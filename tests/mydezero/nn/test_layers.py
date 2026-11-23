"""Linear layer initialization, forward output, gradients, and weight saving."""

import numpy as np
import pytest

from mydezero import Variable, test_mode as inference_mode
from mydezero.layers import Linear, Conv2d, Deconv2d, BatchNorm
from mydezero.models import MLP


@pytest.mark.parametrize('nobias', [False, True])
def test_linear_lazy_initialization_and_gradients(nobias):
    layer = Linear(2, nobias=nobias, dtype=np.float64)
    assert layer.W.data is None
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    y = layer(x)
    weights = layer.W.data.copy()
    y.sum().backward()

    expected = x.data @ weights
    if not nobias:
        expected += layer.b.data
        np.testing.assert_allclose(layer.b.grad.data, [2.0, 2.0])
    else:
        assert layer.b is None
    np.testing.assert_allclose(y.data, expected)
    np.testing.assert_allclose(x.grad.data, np.ones((2, 2)) @ weights.T)
    np.testing.assert_allclose(layer.W.grad.data, x.data.T @ np.ones((2, 2)))
    assert layer.W.data is not None
    assert layer.W.data.dtype == np.float64
    layer(x)
    np.testing.assert_array_equal(layer.W.data, weights)


@pytest.mark.parametrize('nobias', [False, True])
def test_save_load_weights_linear_roundtrip(nobias, tmp_path):
    layer = Linear(2, nobias=nobias, dtype=np.float64, in_size=3)
    path = str(tmp_path / 'linear.npz')
    layer.save_weights(path)

    restored = Linear(2, nobias=nobias, dtype=np.float64, in_size=3)
    restored.load_weights(path)

    np.testing.assert_array_equal(restored.W.data, layer.W.data)
    if not nobias:
        np.testing.assert_array_equal(restored.b.data, layer.b.data)
    else:
        assert restored.b is None


def test_save_load_weights_mlp_restores_predictions(tmp_path):
    model = MLP((4, 3))
    x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
    expected = model(x).data.copy()

    path = str(tmp_path / 'mlp.npz')
    model.save_weights(path)

    restored = MLP((4, 3))
    restored.load_weights(path)
    np.testing.assert_array_equal(restored(x).data, expected)


def test_conv2d_layer_lazy_initialization_and_shapes():
    layer = Conv2d(4, kernel_size=3, pad=1, dtype=np.float32)
    assert layer.W.data is None

    x = Variable(np.random.rand(2, 3, 8, 8).astype(np.float32))
    y = layer(x)

    assert y.shape == (2, 4, 8, 8)
    assert layer.W.shape == (4, 3, 3, 3)
    assert layer.b.shape == (4,)


def test_deconv2d_layer_forward_shape():
    layer = Deconv2d(4, kernel_size=3, pad=1, dtype=np.float32, in_channels=3)
    x = Variable(np.random.rand(2, 3, 5, 5).astype(np.float32))

    y = layer(x)

    assert y.shape == (2, 4, 5, 5)


def test_batch_norm_registers_state_and_restores_inference(tmp_path):
    layer = BatchNorm()
    x = Variable(np.array([[1., 2.], [4., -2.], [5., 3.]], dtype=np.float32))
    y = layer(x)
    assert layer.inputs[0]() is x and layer.outputs[0]() is y
    (y ** 2).sum().backward()
    assert layer.gamma.grad is not None and layer.beta.grad is not None
    assert layer.avg_mean.grad is None and layer.avg_var.grad is None
    layer.gamma.data[:] = [0.5, 2]
    layer.beta.data[:] = [1, -1]

    path = tmp_path / 'batch_norm.npz'
    layer.save_weights(path)
    restored = BatchNorm()
    restored.load_weights(path)

    with np.load(path) as weights:
        assert set(weights.files) == {'avg_mean', 'avg_var', 'gamma', 'beta'}
    with inference_mode():
        np.testing.assert_array_equal(restored(x).data, layer(x).data)
    for name in ('avg_mean', 'avg_var', 'gamma', 'beta'):
        np.testing.assert_array_equal(getattr(restored, name).data, getattr(layer, name).data)


def test_batch_norm_rejects_integer_input_before_initializing_state():
    layer = BatchNorm()
    with pytest.raises(TypeError, match='floating-point input'):
        layer(np.ones((2, 3), dtype=np.int64))
    assert all(param.data is None for param in layer.params())
    assert layer(np.ones((2, 3), dtype=np.float32)).dtype == np.float32
