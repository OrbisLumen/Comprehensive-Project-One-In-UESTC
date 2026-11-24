"""Layer initialization, forward output, gradients, and weight saving."""

import numpy as np
import pytest

from mydezero import Variable, no_grad, test_mode as inference_mode
from mydezero.layers import Linear, Conv2d, Deconv2d, BatchNorm, SelfAttention
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


@pytest.mark.parametrize('layer_class', [Linear, Conv2d, Deconv2d])
@pytest.mark.parametrize('eager', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_weight_initialization_preserves_dtype(layer_class, eager, dtype):
    if layer_class is Linear:
        layer = Linear(4, dtype=dtype, in_size=3 if eager else None)
        shape = (2, 3)
    else:
        layer = layer_class(4, 3, dtype=dtype, in_channels=3 if eager else None)
        shape = (2, 3, 5, 5)
    x = Variable(np.ones(shape, dtype=dtype))
    y = layer(x)
    y.sum().backward()

    assert x.grad.dtype == y.dtype == dtype
    for param in layer.params():
        assert param.dtype == param.grad.dtype == dtype


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


def numerical_gradient(function, data, eps=1e-5):
    gradient = np.zeros_like(data)
    for index in np.ndindex(data.shape):
        above, below = data.copy(), data.copy()
        above[index] += eps
        below[index] -= eps
        with no_grad():
            gradient[index] = (function(above).data - function(below).data) / (2 * eps)
    return gradient


def test_self_attention_forward_matches_numpy_reference():
    rng = np.random.default_rng(10)
    layer = SelfAttention(4, in_size=3, dtype=np.float64)
    x_data = rng.standard_normal((2, 5, 3))

    y = layer(Variable(x_data.copy()))

    assert y.shape == (2, 5, 4)
    assert y.dtype == np.float64
    flat = x_data.reshape(10, 3)
    q = (flat @ layer.q_proj.W.data + layer.q_proj.b.data).reshape(2, 5, 4)
    k = (flat @ layer.k_proj.W.data + layer.k_proj.b.data).reshape(2, 5, 4)
    v = (flat @ layer.v_proj.W.data + layer.v_proj.b.data).reshape(2, 5, 4)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(4)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(y.data, weights @ v)


def test_self_attention_single_token_returns_value_projection():
    rng = np.random.default_rng(11)
    layer = SelfAttention(3, in_size=4, dtype=np.float64)
    x_data = rng.standard_normal((2, 1, 4))

    y = layer(Variable(x_data.copy()))

    expected = layer.v_proj(x_data.reshape(2, 4)).data.reshape(2, 1, 3)
    np.testing.assert_allclose(y.data, expected)


def test_self_attention_registers_parameters_and_gradients():
    layer = SelfAttention(3)
    x = Variable(np.random.default_rng(12).standard_normal((2, 4, 5)).astype(np.float32))

    y = layer(x)
    assert y.shape == (2, 4, 3)
    assert y.dtype == np.float32
    assert len(list(layer.params())) == 6
    y.sum().backward()
    for name in ('q_proj', 'k_proj', 'v_proj'):
        projection = getattr(layer, name)
        assert projection.W.grad is not None
        assert projection.b.grad is not None


def test_self_attention_gradients_match_numerical_gradient():
    rng = np.random.default_rng(13)
    layer = SelfAttention(2, in_size=2, dtype=np.float64)
    x_data = rng.standard_normal((1, 3, 2))
    layer(Variable(x_data.copy()))
    x = Variable(x_data.copy())
    layer(x).sum().backward()

    def objective(values):
        return layer(values).sum()

    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective, x_data),
                               atol=1e-6)

    for name in ('q_proj', 'k_proj', 'v_proj'):
        projection = getattr(layer, name)
        for attribute in ('W', 'b'):
            variable = getattr(projection, attribute)

            def objective_parameter(values, variable=variable):
                original = variable.data
                variable.data = values
                try:
                    return layer(Variable(x_data.copy())).sum()
                finally:
                    variable.data = original

            expected = numerical_gradient(objective_parameter, variable.data.copy())
            np.testing.assert_allclose(variable.grad.data, expected, atol=1e-6)


def test_self_attention_is_batch_independent_and_accepts_noncontiguous_input():
    rng = np.random.default_rng(14)
    layer = SelfAttention(3, in_size=4, dtype=np.float64)
    x_data = rng.standard_normal((2, 3, 4))

    y = layer(Variable(x_data.copy()))
    single = layer(Variable(x_data[1:2].copy()))

    np.testing.assert_allclose(y.data[1:2], single.data)
    np.testing.assert_allclose(layer(Variable(np.asfortranarray(x_data))).data, y.data)


def test_self_attention_rejects_non_sequence_input():
    layer = SelfAttention(2)

    with pytest.raises(ValueError, match=r'\(N, T, C\)'):
        layer(Variable(np.ones((2, 3))))


def test_self_attention_save_load_weights_roundtrip(tmp_path):
    layer = SelfAttention(3, in_size=2)
    x = Variable(np.ones((1, 2, 2), dtype=np.float32))
    expected = layer(x).data.copy()

    path = tmp_path / 'self_attention.npz'
    layer.save_weights(path)
    restored = SelfAttention(3, in_size=2)
    restored.load_weights(path)

    np.testing.assert_array_equal(restored(x).data, expected)
