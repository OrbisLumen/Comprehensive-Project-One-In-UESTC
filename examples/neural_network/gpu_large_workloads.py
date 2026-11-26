"""Matched large-model GPU inference and training experiments for the notebook."""

from contextlib import contextmanager, nullcontext
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import cupy as cp
import numpy as np

from mydezero import Variable, no_grad, test_mode, using_config
import mydezero.functions as F
import mydezero.functions_conv as FC
import mydezero.models as models
from mydezero.models import ResNet50, VGG16
from mydezero.optimizers import Adam
import mydezero.optimizers as optimizers


_layout_eligible = optimizers._adam_can_fuse


def c_only_fusion(xp, *buffers):
    return all(a.flags.c_contiguous for a in buffers) and _layout_eligible(xp, *buffers)


def reference_pool(self, x):
    self.direct_gpu = False
    columns = FC.im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
    n, c, kh, kw, oh, ow = columns.shape
    columns = columns.reshape(n, c, kh * kw, oh, ow)
    self.indexes = columns.argmax(axis=2)
    return columns.max(axis=2)


def reference_loss(self, x, labels):
    self.gpu_probabilities = None
    shifted = x - x.max(axis=1, keepdims=True)
    log_probabilities = shifted - F.utils.logsumexp(shifted, axis=1)
    return -log_probabilities[np.arange(x.shape[0]), labels.ravel()].mean()


def reference_batchnorm_backward(self, gy):
    x, gamma, _ = self.inputs
    gbeta = gy.sum(axis=self.axes)
    ggamma = (gy * self.normalized).sum(axis=self.axes)
    if self.train:
        gy = gy - (gbeta.reshape(self.channel_shape)
                   + self.normalized * ggamma.reshape(self.channel_shape)) / self.count
    return gy * gamma.reshape(self.channel_shape) * self.inv_std, ggamma, gbeta


def reference_average_pool(x):
    return F.sum(x, axis=(2, 3)) / (x.shape[2] * x.shape[3])


@contextmanager
def mode(name):
    support = c_only_fusion if name in ('Flat GEMM', 'cuDNN') else _layout_eligible
    with using_config('use_cudnn', name.startswith('cuDNN')), \
            patch.object(optimizers, '_adam_can_fuse', support):
        if name in ('Original', 'FP32 reference'):
            with patch.object(FC, '_gpu_pointwise', lambda *_: False), \
                    patch.object(FC.Pooling, 'forward', reference_pool), \
                    patch.object(F.SoftmaxCrossEntropy, 'forward', reference_loss):
                if name == 'Original':
                    with patch.object(F.BatchNorm, 'backward', reference_batchnorm_backward), \
                            patch.object(models, '_global_average_pooling_2d', reference_average_pool):
                        yield
                else:
                    yield
        else:
            yield


def measure(call, rounds=5, calls=5):
    for _ in range(3):
        call()
    cp.cuda.runtime.deviceSynchronize()
    samples = []
    medians = []
    for _ in range(rounds):
        current = []
        for _ in range(calls):
            start = perf_counter()
            call()
            cp.cuda.runtime.deviceSynchronize()
            current.append((perf_counter() - start) * 1000)
        samples.extend(current)
        medians.append(median(current))
    return dict(p50=median(medians), p95=float(np.percentile(samples, 95)),
                round_medians=medians, samples=samples)


def measure_paired(calls, rounds=5, samples=5):
    history = [[] for _ in calls]
    for context, call in calls:
        with context():
            for _ in range(3):
                call()
    cp.cuda.runtime.deviceSynchronize()
    for r in range(rounds):
        current = [[] for _ in calls]
        for j in range(samples):
            for k in range(len(calls)):
                i = (r + j + k) % len(calls)
                context, call = calls[i]
                with context():
                    start = perf_counter()
                    call()
                    cp.cuda.runtime.deviceSynchronize()
                    current[i].append((perf_counter() - start) * 1000)
        for values, row in zip(history, current):
            values.append(row)
    return [dict(p50=median(median(row) for row in group),
                 p95=float(np.percentile(np.concatenate(group), 95)),
                 round_medians=[median(row) for row in group], samples=np.concatenate(group).tolist())
            for group in history]


def load(model_class, archive):
    model = model_class()
    model.load_weights(archive)
    model.to_gpu()
    return model


def gradients(model):
    parameters = {}
    model._flatten_params(parameters)
    return {name: cp.asnumpy(p.grad.data) for name, p in parameters.items() if p.grad is not None}


def gradient_error(actual, expected):
    squared_error, squared_reference = 0.0, 0.0
    if actual.keys() != expected.keys():
        raise AssertionError('Different gradient parameter keys')
    for name in actual:
        left, right = actual[name].astype(np.float64), expected[name].astype(np.float64)
        squared_error += np.square(left - right).sum()
        squared_reference += np.square(right).sum()
    return float(np.sqrt(squared_error / max(squared_reference, np.finfo(float).tiny)))


def torch_model(source, name, torch):
    """Copy the source architecture, including original-MSRA downsampling."""
    nn = torch.nn

    def conv(layer):
        result = nn.Conv2d(layer.W.shape[1], layer.W.shape[0], layer.W.shape[2:],
                           stride=layer.stride, padding=layer.pad, bias=layer.b is not None)
        result.weight.data.copy_(torch.from_numpy(cp.asnumpy(layer.W.data)))
        if layer.b is not None:
            result.bias.data.copy_(torch.from_numpy(cp.asnumpy(layer.b.data)))
        return result

    def linear(layer):
        result = nn.Linear(layer.W.shape[0], layer.W.shape[1])
        result.weight.data.copy_(torch.from_numpy(cp.asnumpy(layer.W.data).T))
        result.bias.data.copy_(torch.from_numpy(cp.asnumpy(layer.b.data)))
        return result

    def bn(layer):
        result = nn.BatchNorm2d(layer.gamma.size, eps=2e-5, momentum=0.1)
        for target, value in ((result.weight, layer.gamma), (result.bias, layer.beta),
                              (result.running_mean, layer.avg_mean), (result.running_var, layer.avg_var)):
            target.data.copy_(torch.from_numpy(cp.asnumpy(value.data)))
        return result

    class Bottleneck(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.conv1, self.bn1 = conv(block.conv1), bn(block.bn1)
            self.conv2, self.bn2 = conv(block.conv2), bn(block.bn2)
            self.conv3, self.bn3 = conv(block.conv3), bn(block.bn3)
            self.shortcut = (nn.Sequential(conv(block.conv4), bn(block.bn4))
                             if hasattr(block, 'conv4') else nn.Identity())

        def forward(self, x):
            y = self.bn1(self.conv1(x)).relu()
            y = self.bn2(self.conv2(y)).relu()
            y = self.bn3(self.conv3(y))
            return (y + self.shortcut(x)).relu()

    class Network(nn.Module):
        def __init__(self, source):
            super().__init__()
            if name == 'ResNet50':
                self.stem = nn.Sequential(conv(source.conv1), bn(source.bn1), nn.ReLU(), nn.MaxPool2d(3, 2))
                blocks = []
                for stage_name in ('res2', 'res3', 'res4', 'res5'):
                    stage = getattr(source, stage_name)
                    blocks.extend(Bottleneck(getattr(stage, key)) for key in stage._forward)
                self.blocks = nn.Sequential(*blocks)
                self.classifier = linear(source.fc6)
            else:
                layers = []
                for stage, count in enumerate((2, 2, 3, 3, 3), 1):
                    for index in range(1, count + 1):
                        layers.extend((conv(getattr(source, f'conv{stage}_{index}')), nn.ReLU()))
                    layers.append(nn.MaxPool2d(2, 2))
                self.blocks = nn.Sequential(*layers)
                self.classifier = nn.Sequential(linear(source.fc6), nn.ReLU(), nn.Dropout(0.5),
                                                linear(source.fc7), nn.ReLU(), nn.Dropout(0.5), linear(source.fc8))

        def forward(self, x):
            if name == 'ResNet50':
                x = self.blocks(self.stem(x)).mean(dim=(2, 3))
            else:
                x = self.blocks(x).flatten(1)
            return self.classifier(x)

    return Network(source).cuda()


def run(weights_dir, process_index=0, output_path=None, layout_followup=False):
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    modes = ('Original', 'FP32 reference', 'Flat GEMM', 'cuDNN')
    cases = ((ResNet50, 'resnet50.npz', (1, 4, 16)), (VGG16, 'vgg16.npz', (1, 4, 8)))
    if layout_followup:
        modes = ('Fortran Adam', 'cuDNN+Fortran Adam')
        cases = ((VGG16, 'vgg16.npz', (1, 4, 8)),)
    for model_class, filename, batches in cases:
        archive = Path(weights_dir) / filename
        for batch in batches:
            values = np.random.default_rng(39).standard_normal((batch, 3, 224, 224), dtype=np.float32)
            x, labels = cp.asarray(values), cp.arange(batch, dtype=cp.int64) % 1000
            source = load(model_class, archive)
            with no_grad(), test_mode(), mode('FP32 reference'):
                reference_logits = cp.asnumpy(source(x).data)
            matched = torch_model(source, model_class.__name__, torch).eval()
            tx = torch.from_numpy(values).cuda()
            with no_grad(), test_mode(), torch.inference_mode():
                for name in modes:
                    with mode(name):
                        actual = cp.asnumpy(source(x).data)
                    np.testing.assert_allclose(actual, reference_logits, rtol=1e-3, atol=3e-4)
                    np.testing.assert_array_equal(actual.argmax(axis=1), reference_logits.argmax(axis=1))
                np.testing.assert_allclose(matched(tx).cpu().numpy(), reference_logits, rtol=1e-3, atol=4e-4)
                inference_calls = [(lambda n=name: mode(n), lambda: source(x)) for name in modes]
                inference_calls.append((nullcontext, lambda: matched(tx)))
                inference_values = dict(zip((*modes, 'PyTorch'), measure_paired(inference_calls)))
            del matched, tx, inference_calls
            torch.cuda.empty_cache()
            with mode('FP32 reference'):
                cp.random.seed(73)
                source.cleargrads()
                reference_loss = F.softmax_cross_entropy(source(x), labels)
                reference_loss.backward()
                reference_loss_value = float(reference_loss.data)
                reference_gradient = gradients(source)
            del source, reference_loss
            case_rows = []
            offset = process_index % len(modes)
            for name in modes[offset:] + modes[:offset]:
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                model = load(model_class, archive)
                optimizer = Adam(lr=1e-4, fused=name not in ('Original', 'FP32 reference')).setup(model)
                row = dict(model=model_class.__name__, batch=batch, mode=name, process=process_index)
                with mode(name):
                    with no_grad(), test_mode():
                        started = perf_counter()
                        output = model(x).data
                        cp.cuda.runtime.deviceSynchronize()
                        row['first_inference_ms'] = (perf_counter() - started) * 1000
                        logits = cp.asnumpy(output)
                        row['logit_dtype'] = str(logits.dtype)
                        np.testing.assert_allclose(logits, reference_logits, rtol=1e-3, atol=3e-4)
                        np.testing.assert_array_equal(logits.argmax(axis=1), reference_logits.argmax(axis=1))
                        row['inference'] = inference_values[name]
                    model.cleargrads()
                    cp.random.seed(73)
                    first_loss = F.softmax_cross_entropy(model(x), labels)
                    first_loss.backward()
                    loss_value = float(first_loss.data)
                    current_gradient = gradients(model)
                    row['gradient_dtypes'] = sorted({str(a.dtype) for a in current_gradient.values()})
                    row['gradient_relative_l2'] = gradient_error(current_gradient, reference_gradient)
                    row['first_training_loss'] = loss_value
                    np.testing.assert_allclose(loss_value, reference_loss_value, rtol=1e-3, atol=2e-4)
                    del current_gradient, first_loss, output

                    def train():
                        model.cleargrads()
                        loss = F.softmax_cross_entropy(model(x), labels)
                        loss.backward()
                        optimizer.update()
                        return loss

                    cp.random.seed(971)
                    if row['gradient_relative_l2'] < 0.005 or name == 'Original':
                        row['training'] = measure(train, rounds=3, calls=3)
                    else:
                        row['training'] = None
                        row['rejected'] = 'First-step gradient relative L2 exceeds 0.005'
                    row['pool_retained_mib'] = cp.get_default_memory_pool().total_bytes() / (1024 ** 2)
                print(model_class.__name__, batch, name,
                      'infer/train', round(row['inference']['p50'], 3),
                      None if row['training'] is None else round(row['training']['p50'], 3),
                      'gradient error', row['gradient_relative_l2'], flush=True)
                case_rows.append(row)
                del model, optimizer
                cp.cuda.runtime.deviceSynchronize()
            reference_gradient = None
            # Build the matched PyTorch architecture from the same official checkpoint.
            cp.get_default_memory_pool().free_all_blocks()
            source = load(model_class, archive)
            matched = torch_model(source, model_class.__name__, torch)
            del source
            cp.get_default_memory_pool().free_all_blocks()
            tx, target = torch.from_numpy(values).cuda(), torch.arange(batch, device='cuda') % 1000
            inference = inference_values['PyTorch']
            matched.train()
            optimizer = torch.optim.Adam(matched.parameters(), lr=1e-4)

            def train_torch():
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(matched(tx), target)
                loss.backward()
                optimizer.step()

            training = measure(train_torch, rounds=3, calls=3)
            case_rows.append(dict(model=model_class.__name__, batch=batch, mode='PyTorch',
                                  process=process_index, inference=inference, training=training))
            print(model_class.__name__, batch, 'PyTorch infer/train',
                  round(inference['p50'], 3), round(training['p50'], 3), flush=True)
            rows.extend(case_rows)
            if output_path is not None:
                Path(output_path).write_text(json.dumps(rows, indent=2), encoding='utf-8')
            del matched, optimizer, tx, target, x, labels
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process', type=int, default=0)
    parser.add_argument('--layout-followup', action='store_true')
    args = parser.parse_args()
    run(args.weights, args.process, args.output, args.layout_followup)
