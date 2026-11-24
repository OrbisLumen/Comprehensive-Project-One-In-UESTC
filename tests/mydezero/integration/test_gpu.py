"""Real CuPy execution, transfers, CNN gradients, and optimizer updates."""

import numpy as np
import pytest

from mydezero import Layer, Parameter, Variable, cuda, test_mode as inference_mode
import mydezero.functions as F
from mydezero.layers import BatchNorm
from mydezero.models import MLP
import mydezero.optimizers as optimizers
from mydezero.optimizers import Adam, MomentumSGD


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f'CUDA device unavailable: {error}')
    if count == 0:
        pytest.skip('No CUDA device')
    # Compilation/library failures on an available GPU must fail, not skip.
    cupy.testing.assert_array_equal(cupy.arange(3) + 1, cupy.array([1, 2, 3]))
    return cupy


def test_device_transfer_and_broadcast_backward(cp):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = Variable(data.copy())
    x.to_gpu()
    assert isinstance(x.data, cp.ndarray)
    assert cuda.get_array_module(x) is cp
    bias = Variable(cp.array([1., 2., 3.], dtype=cp.float32))
    loss = ((x + bias) ** 2).sum()
    loss.backward()
    assert isinstance(x.grad.data, cp.ndarray)
    np.testing.assert_allclose(cuda.as_numpy(x.grad), 2 * (data + [1, 2, 3]))
    np.testing.assert_allclose(cuda.as_numpy(bias.grad), (2 * (data + [1, 2, 3])).sum(axis=0))
    x.to_cpu()
    assert cuda.get_array_module(x) is np
    np.testing.assert_array_equal(x.data, data)


@pytest.mark.parametrize('operation', [F.conv2d, F.deconv2d])
@pytest.mark.parametrize('stride,pad', [(1, 0), ((2, 1), (1, 0))])
def test_convolution_forward_and_all_gradients_match_cpu(cp, operation, stride, pad):
    rng = np.random.default_rng(7)
    weight_shape = (3, 2, 2, 3) if operation is F.conv2d else (2, 3, 2, 3)
    arrays = [rng.normal(size=(2, 2, 4, 5)), rng.normal(size=weight_shape), rng.normal(size=3)]
    cpu = [Variable(a.astype(np.float32)) for a in arrays]
    gpu = [Variable(cp.asarray(v.data)) for v in cpu]
    expected = operation(*cpu, stride=stride, pad=pad)
    actual = operation(*gpu, stride=stride, pad=pad)
    (expected ** 2).sum().backward()
    (actual ** 2).sum().backward()
    assert isinstance(actual.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual.data), expected.data, rtol=2e-5, atol=2e-5)
    for host, device in zip(cpu, gpu):
        assert isinstance(device.grad.data, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(device.grad.data), host.grad.data, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('operation', [F.pooling, F.average_pooling])
def test_pooling_forward_and_gradient_match_cpu(cp, operation):
    data = np.random.default_rng(8).normal(size=(2, 2, 4, 5)).astype(np.float32)
    cpu, gpu = Variable(data), Variable(cp.asarray(data))
    expected = operation(cpu, (2, 3), stride=(1, 2), pad=(1, 0))
    actual = operation(gpu, (2, 3), stride=(1, 2), pad=(1, 0))
    (expected ** 2).sum().backward()
    (actual ** 2).sum().backward()
    assert isinstance(actual.data, cp.ndarray)
    assert isinstance(gpu.grad.data, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual.data), expected.data, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(gpu.grad.data), cpu.grad.data, atol=1e-6)


def test_gpu_classifier_training_and_model_transfer(cp):
    x = cp.array([[-2.], [-1.], [1.], [2.]], dtype=cp.float32)
    labels = cp.array([0, 0, 1, 1], dtype=cp.int64)
    model = MLP((2,))
    model.to_gpu()
    model(x)  # Lazy weights must also initialize on GPU.
    model.l0.W.data[:] = 0
    model.l0.b.data[:] = 0
    optimizer = MomentumSGD(lr=0.1, momentum=0.5).setup(model)
    initial = F.softmax_cross_entropy(model(x), labels).data.item()
    for _ in range(30):
        model.cleargrads()
        loss = F.softmax_cross_entropy(model(x), labels)
        loss.backward()
        for param in model.params():
            assert isinstance(param.data, cp.ndarray)
            assert isinstance(param.grad.data, cp.ndarray)
        optimizer.update()
    logits = model(x)
    assert F.softmax_cross_entropy(logits, labels).data.item() < initial * 0.25
    cp.testing.assert_array_equal(logits.data.argmax(axis=1), labels)
    expected = cp.asnumpy(logits.data)
    model.to_cpu()
    assert all(isinstance(p.data, np.ndarray) for p in model.params())
    np.testing.assert_allclose(model(cp.asnumpy(x)).data, expected, rtol=1e-5, atol=1e-6)


def test_batch_norm_gradients_and_running_statistics_match_cpu(cp):
    data = np.random.default_rng(11).normal(size=(2, 3, 2, 4)).astype(np.float32)
    x = Variable(data)
    device_x = Variable(cp.asarray(data))
    cpu, gpu = BatchNorm(), BatchNorm()
    expected = cpu(x)
    actual = gpu(device_x)
    (expected ** 2).sum().backward()
    (actual ** 2).sum().backward()

    np.testing.assert_allclose(cp.asnumpy(actual.data), expected.data, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(device_x.grad.data), x.grad.data, atol=1e-5)
    for name in ('gamma', 'beta'):
        host_grad = getattr(cpu, name).grad.data
        device_grad = getattr(gpu, name).grad.data
        np.testing.assert_allclose(cp.asnumpy(device_grad), host_grad, atol=1e-5)
    gpu.to_cpu()
    for name in ('avg_mean', 'avg_var'):
        np.testing.assert_allclose(getattr(gpu, name).data, getattr(cpu, name).data, atol=1e-6)
    with inference_mode():
        np.testing.assert_allclose(gpu(x).data, cpu(x).data, atol=1e-6)


def test_fused_adam_matches_unfused_cuda_updates(cp, monkeypatch):
    fused_kernel = optimizers._adam_fp32_update
    calls = []

    def count_update(*args):
        calls.append(args[0].shape)
        fused_kernel(*args)

    monkeypatch.setattr(optimizers, '_adam_fp32_update', count_update)

    def scale_gradients(params):
        for param in params:
            param.grad.data *= 0.5

    pairs = []
    for fused in (False, True):
        layer = Layer()
        layer.W = Parameter(cp.ones((2, 2), dtype=cp.float32))
        layer.b = Parameter(cp.zeros(2, dtype=cp.float32))
        optimizer = Adam(lr=0.01, fused=fused).setup(layer)
        optimizer.add_hook(scale_gradients)
        pairs.append((layer, optimizer))

    for step in range(8):
        for layer, optimizer in pairs:
            layer.W.grad = Variable(cp.full((2, 2), step - 2, dtype=cp.float32))
            layer.b.grad = None if step % 2 else Variable(cp.array([0.2, -0.5], dtype=cp.float32))
            optimizer.update()
        (standard, unfused), (fused, optimized) = pairs
        for left, right in ((standard.W, fused.W), (standard.b, fused.b)):
            assert unfused.ts[id(left)] == optimized.ts[id(right)]
            cp.testing.assert_allclose(left.data, right.data, rtol=1e-6, atol=1e-7)
            for left_moments, right_moments in ((unfused.ms, optimized.ms), (unfused.vs, optimized.vs)):
                cp.testing.assert_allclose(left_moments[id(left)], right_moments[id(right)],
                                           rtol=1e-6, atol=1e-7)

    assert calls.count((2, 2)) == 8
    assert calls.count((2,)) == 4


def test_fused_adam_matches_fp32_power_rounding(cp):
    rng = np.random.default_rng(522)
    state = []
    for fused in (False, True):
        param = Parameter(cp.ones(256, dtype=cp.float32))
        optimizer = Adam(fused=fused)
        state.append((param, optimizer))

    for _ in range(4):
        grad = rng.standard_normal(256, dtype=np.float32) * np.float32(0.001)
        for param, optimizer in state:
            param.grad = Variable(cp.asarray(grad))
            optimizer.update_one(param)

    cp.testing.assert_array_equal(*(optimizer.vs[id(param)] for param, optimizer in state))


@pytest.mark.parametrize('layout', ['fp64_weight', 'fp64_gradient', 'strided_weight', 'overlapping_gradient'])
def test_fused_adam_preserves_fallback(cp, monkeypatch, layout):
    def reject_kernel(*args):
        pytest.fail("Unsupported arrays must use the standard update")

    monkeypatch.setattr(optimizers, '_adam_fp32_update', reject_kernel)
    variants = []
    for fused in (False, True):
        values = cp.arange(8, dtype=cp.float32)
        weight, grad = values[:4].copy(), values[4:].copy()
        if layout == 'fp64_weight':
            weight = weight.astype(cp.float64)
        elif layout == 'fp64_gradient':
            grad = grad.astype(cp.float64)
        elif layout == 'strided_weight':
            weight = values[::2]
        else:
            weight, grad = values[:4], values[2:6]
        param = Parameter(weight)
        param.grad = Variable(grad)
        optimizer = Adam(fused=fused)
        optimizer.update_one(param)
        variants.append(param.data)
    cp.testing.assert_array_equal(*variants)
