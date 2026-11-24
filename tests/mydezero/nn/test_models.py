"""Recursive parameter collection and gradient clearing on models."""

import numpy as np
import pytest

from mydezero import Variable, no_grad, test_mode as inference_mode
import mydezero.functions as F
from mydezero.layers import Linear
from mydezero.models import MLP, Sequential, VGG16
from mydezero.models import BottleneckA, BottleneckB, ResNet, ResNet50, ResNet101, ResNet152
from mydezero.models import _global_average_pooling_2d


def test_nested_model_collects_parameters(regression_model):
    model = regression_model

    assert set(model.params()) == {model.linear.W, model.linear.b, model.unused}


def test_nested_model_clears_all_gradients(regression_model):
    model = regression_model
    for param in model.params():
        param.grad = Variable(np.ones_like(param.data))

    model.cleargrads()

    assert all(param.grad is None for param in model.params())


def test_sequential_applies_layers_in_order_and_backpropagates():
    first = Linear(2, in_size=2, dtype=np.float64)
    second = Linear(1, in_size=2, dtype=np.float64)
    first.W.data[:] = [[1, 2], [3, 4]]
    first.b.data[:] = [1, -1]
    second.W.data[:] = [[2], [-1]]
    second.b.data[:] = [0.5]
    model = Sequential(first, second)
    x = Variable(np.array([[1.0, 2.0]]))

    y = model(x)
    y.sum().backward()

    np.testing.assert_allclose(y.data, [[7.5]])
    np.testing.assert_allclose(x.grad.data, [[0, 2]])
    np.testing.assert_allclose(first.W.grad.data, [[2, -1], [4, -2]])
    np.testing.assert_allclose(first.b.grad.data, [2, -1])
    np.testing.assert_allclose(second.W.grad.data, [[8], [9]])
    np.testing.assert_allclose(second.b.grad.data, [1])
    assert model.l0 is first and model.l1 is second
    assert set(model.params()) == {first.W, first.b, second.W, second.b}
    model.cleargrads()
    assert all(param.grad is None for param in model.params())


def test_empty_sequential_returns_input():
    model = Sequential()
    x = Variable(np.array([[1.0, 2.0]]))

    assert model(x) is x
    assert list(model.params()) == []


@pytest.mark.parametrize('activation', [None, F.tanh], ids=['default-sigmoid', 'tanh'])
def test_mlp_initializes_layers_and_activates_only_hidden_outputs(activation):
    model = MLP((3, 1)) if activation is None else MLP((3, 1), activation)
    assert all(layer.W.data is None for layer in model.layers)
    x = Variable(np.array([[1.0, -2.0], [0.5, 1.0]]))
    model(x)
    assert [layer.W.shape for layer in model.layers] == [(2, 3), (3, 1)]
    model.l0.W.data[:] = [[0.2, -0.3, 0.4], [0.5, 0.6, -0.7]]
    model.l0.b.data[:] = [0.1, -0.2, 0.3]
    model.l1.W.data[:] = [[1.0], [-2.0], [0.5]]
    model.l1.b.data[:] = [3.0]

    y = model(x)
    z = x.data @ model.l0.W.data + model.l0.b.data
    hidden = 1 / (1 + np.exp(-z)) if activation is None else np.tanh(z)
    expected = hidden @ model.l1.W.data + model.l1.b.data
    np.testing.assert_allclose(y.data, expected, rtol=1e-6)
    y.sum().backward()
    derivative = hidden * (1 - hidden) if activation is None else 1 - hidden ** 2
    hidden_grad = derivative * model.l1.W.data.T
    np.testing.assert_allclose(model.l0.W.grad.data, x.data.T @ hidden_grad, rtol=1e-6)
    np.testing.assert_allclose(model.l1.W.grad.data, hidden.sum(axis=0)[:, None])
    assert set(model.params()) == {model.l0.W, model.l0.b, model.l1.W, model.l1.b}
    model.cleargrads()
    assert all(param.grad is None for param in model.params())


def test_single_layer_mlp_does_not_apply_activation():
    def unexpected_activation(x):
        pytest.fail('A single-layer MLP has no hidden activation.')

    model = MLP((1,), activation=unexpected_activation)
    x = Variable(np.array([[2.0]]))
    model(x)
    model.l0.W.data[:] = 3
    model.l0.b.data[:] = -1

    np.testing.assert_allclose(model(x).data, [[5]])


def test_model_plot_forwards_input_and_filename(monkeypatch):
    from mydezero import utils
    from mydezero.models import MLP

    model = MLP((2,))
    x = np.zeros((3, 4), dtype=np.float32)
    calls = []
    sentinel = object()

    def render(output, verbose, to_file):
        calls.append((output, verbose, to_file))
        return sentinel

    monkeypatch.setattr(utils, 'plot_dot_graph', render)
    assert model.plot(x, to_file='custom.png') is sentinel
    output, verbose, filename = calls[0]
    assert output.shape == (3, 2) and verbose is True and filename == 'custom.png'


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_vgg16_preprocess_centers_bgr_channels(dtype):
    Image = pytest.importorskip('PIL.Image')
    image = Image.new('RGB', (2, 3), (10, 20, 30))

    result = VGG16.preprocess(image, size=None, dtype=dtype)

    assert result.shape == (3, 3, 2)
    assert result.dtype == dtype
    expected = np.array([-73.939, -96.779, -113.68], dtype=dtype)
    np.testing.assert_allclose(result, np.broadcast_to(expected[:, None, None], result.shape))
    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_vgg16_preprocess_rejects_integer_dtype():
    Image = pytest.importorskip('PIL.Image')

    with pytest.raises(TypeError, match='floating-point dtype'):
        VGG16.preprocess(Image.new('RGB', (2, 3)), dtype=np.uint8)


@pytest.mark.parametrize('model_class, stages', [
    (ResNet50, (3, 4, 6, 3)),
    (ResNet101, (3, 4, 23, 3)),
    (ResNet152, (3, 8, 36, 3)),
])
def test_resnet_variants_register_dezero_block_names(model_class, stages):
    model = model_class()
    for i, count in enumerate(stages, start=2):
        stage = getattr(model, f'res{i}')
        assert stage._forward == ['a'] + [f'b{j}' for j in range(1, count)]

    weights = {}
    model._flatten_params(weights)
    assert weights['res2/a/conv4/W'] is model.res2.a.conv4.W
    assert weights['res2/a/bn1/avg_mean'] is model.res2.a.bn1.avg_mean
    assert weights['res5/b2/bn3/avg_var'] is model.res5.b2.bn3.avg_var


def test_resnet_rejects_unsupported_depth():
    with pytest.raises(ValueError, match='50, 101,.*152'):
        ResNet(18)


@pytest.mark.parametrize('downsample_fb', [False, True])
def test_resnet_bottlenecks_downsample_and_backpropagate(downsample_fb):
    x = Variable(np.random.default_rng(10).normal(size=(2, 4, 5, 7)).astype(np.float32))
    projection = BottleneckA(2, 8, stride=2, downsample_fb=downsample_fb)
    identity = BottleneckB(8, 2)

    y = identity(projection(x))
    y.sum().backward()

    assert y.shape == (2, 8, 3, 4)
    assert x.grad.shape == x.shape
    assert np.isfinite(x.grad.data).all()
    assert projection.conv4.W.grad is not None and identity.conv3.W.grad is not None


def test_resnet50_inference_returns_finite_logits():
    model = ResNet50()
    x = np.ones((1, 3, 32, 32), dtype=np.float32)

    with inference_mode(), no_grad():
        logits = model(x)

    assert logits.shape == (1, 1000)
    assert np.isfinite(logits.data).all()
    assert logits.creator is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_resnet_global_average_pooling_matches_previous_forward_and_backward(dtype):
    data = np.random.default_rng(14).normal(size=(2, 3, 5, 4)).astype(dtype)
    weights = np.array([[1., -3., 2.], [0.5, 0., -4.]], dtype=dtype)
    x, reference = Variable(data), Variable(data.copy())

    result = _global_average_pooling_2d(x)
    expected = F.average_pooling(reference, (5, 4)).reshape(2, 3)
    (result * weights).sum().backward()
    (expected * weights).sum().backward()

    assert result.shape == (2, 3)
    assert result.dtype == x.grad.dtype == dtype
    np.testing.assert_allclose(result.data, expected.data, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(x.grad.data, reference.grad.data, rtol=1e-6)


def test_global_average_pooling_count_exceeding_fp16_range():
    x = Variable(np.full((1, 1, 256, 256), 0.125, dtype=np.float16))
    y = _global_average_pooling_2d(x)
    y.backward()
    np.testing.assert_array_equal(y.data, [[0.125]])
    np.testing.assert_allclose(x.grad.data, np.full(x.shape, 1 / 65536))
