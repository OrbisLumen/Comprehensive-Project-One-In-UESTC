"""Channel normalization and gradients in training and inference."""

import numpy as np
import pytest

from mydezero import Variable, no_grad, using_config
from mydezero.functions import batch_norm


@pytest.mark.parametrize('create_graph', [False, True])
def test_batch_norm_count_exceeding_fp16_range(create_graph):
    x = Variable(np.zeros((65536, 1), dtype=np.float16))
    gamma = np.ones(1, dtype=np.float16)
    beta = np.zeros(1, dtype=np.float16)
    y = batch_norm(x, gamma, beta, beta.copy(), gamma.copy())
    y.grad = Variable(np.full(x.shape, 0.125, dtype=np.float16))
    y.backward(create_graph=create_graph)
    np.testing.assert_array_equal(x.grad.data, np.zeros(x.shape))


@pytest.mark.parametrize('shape', [(2, 2), (3, 2), (2, 2, 2, 3)])
@pytest.mark.parametrize('train', [True, False], ids=['train', 'inference'])
def test_batch_norm_forward_and_numerical_gradients(shape, train):
    rng = np.random.default_rng(42)
    data = rng.normal(size=shape)
    weights = rng.normal(size=shape)
    scale = np.array([0.5, 1.7])
    bias = np.array([0.1, -0.4])
    initial_mean = np.array([0.3, -0.2])
    initial_var = np.array([0.8, 1.5])
    avg_mean, avg_var = initial_mean.copy(), initial_var.copy()
    x, gamma, beta = map(Variable, (data, scale, bias))
    axes = (0,) if len(shape) == 2 else (0, 2, 3)
    channel_shape = (1, 2) + (1,) * (len(shape) - 2)
    count = data.size // 2
    batch_mean, batch_var = data.mean(axis=axes), data.var(axis=axes)

    with using_config('train', train):
        y = batch_norm(x, gamma, beta, avg_mean, avg_var)
    with using_config('train', not train):
        (y * weights).sum().backward()

    mean, var = (batch_mean, batch_var) if train else (initial_mean, initial_var)
    normalized = (data - mean.reshape(channel_shape)) / np.sqrt(var.reshape(channel_shape) + 2e-5)
    expected = normalized * scale.reshape(channel_shape) + bias.reshape(channel_shape)
    np.testing.assert_allclose(y.data, expected)
    if train:
        np.testing.assert_allclose(avg_mean, 0.9 * initial_mean + 0.1 * batch_mean)
        correction = count / max(count - 1, 1)
        np.testing.assert_allclose(avg_var, 0.9 * initial_var + 0.1 * correction * batch_var)
    else:
        np.testing.assert_array_equal(avg_mean, initial_mean)
        np.testing.assert_array_equal(avg_var, initial_var)

    def objective(values):
        with no_grad(), using_config('train', train):
            result = batch_norm(*values, initial_mean.copy(), initial_var.copy())
            return (result.data * weights).sum()

    for parameter_index, parameter in enumerate((x, gamma, beta)):
        numerical = np.empty_like(parameter.data)
        for index in np.ndindex(parameter.shape):
            values = [array.copy() for array in (data, scale, bias)]
            values[parameter_index][index] += 1e-5
            above = objective(values)
            values[parameter_index][index] -= 2e-5
            numerical[index] = (above - objective(values)) / 2e-5
        np.testing.assert_allclose(parameter.grad.data, numerical, atol=1e-8)


def test_batch_norm_handles_one_value_per_channel_in_training():
    x = Variable(np.array([[2, -4]], dtype=np.float32))
    avg_mean = np.zeros(2, dtype=np.float32)
    avg_var = np.ones(2, dtype=np.float32)
    y = batch_norm(x, np.ones_like(avg_mean), np.zeros_like(avg_mean), avg_mean, avg_var)
    y.sum().backward()

    assert y.dtype == x.grad.dtype == np.float32
    np.testing.assert_array_equal(y.data, np.zeros_like(x.data))
    np.testing.assert_array_equal(x.grad.data, np.zeros_like(x.data))
    np.testing.assert_allclose(avg_mean, [0.2, -0.4])
    np.testing.assert_allclose(avg_var, [0.9, 0.9])


def test_batch_norm_updates_running_statistics_under_no_grad_in_training():
    avg_mean = np.array([1.0])
    avg_var = np.array([11.0])
    gamma, beta = np.array([2.0]), np.array([1.0])

    with using_config('train', True), no_grad():
        first = batch_norm(np.array([[0.0], [4.0]]), gamma, beta, avg_mean, avg_var,
                           decay=0.25, eps=1.0)
        second = batch_norm(np.array([[2.0], [4.0]]), gamma, beta, avg_mean, avg_var,
                            decay=0.25, eps=1.0)

    np.testing.assert_allclose(first.data, [[1 - 4 / np.sqrt(5)], [1 + 4 / np.sqrt(5)]])
    np.testing.assert_allclose(second.data, [[1 - np.sqrt(2)], [1 + np.sqrt(2)]])
    np.testing.assert_allclose(avg_mean, [2.6875])
    np.testing.assert_allclose(avg_var, [3.6875])
    assert first.creator is None and second.creator is None


def test_batch_norm_training_second_derivative_matches_finite_difference():
    data = np.array([[1.0], [2.0], [-2.0]])
    direction = np.array([[0.5], [0.1], [-0.2]])
    weights = np.array([[0.3], [-1.0], [0.5]])

    def gradient(values, create_graph=False):
        x = Variable(values)
        y = batch_norm(x, np.ones(1), np.zeros(1), np.zeros(1), np.ones(1))
        (y * weights).sum().backward(create_graph=create_graph)
        return x, x.grad

    x, first = gradient(data, create_graph=True)
    x.cleargrad()
    (first * direction).sum().backward()
    eps = 1e-5
    above = gradient(data + eps * direction)[1].data
    below = gradient(data - eps * direction)[1].data
    np.testing.assert_allclose(x.grad.data, (above - below) / (2 * eps), atol=1e-8)


def test_batch_norm_inference_mixed_derivative_uses_forward_statistics():
    x = Variable(np.array([[1.0, 3.0], [2.0, -1.0]]))
    gamma = Variable(np.array([1.5, 0.5]))
    avg_mean = np.array([0.5, -0.5])
    avg_var = np.array([0.25, 4.0])
    inv_std = 1 / np.sqrt(avg_var + 2e-5)
    expected_gamma_grad = ((x.data - avg_mean) * inv_std).sum(axis=0)
    with using_config('train', False):
        loss = batch_norm(x, gamma, np.zeros(2), avg_mean, avg_var).sum()

    avg_mean[:] = 20
    avg_var[:] = 10
    loss.backward(create_graph=True)
    gamma_grad = gamma.grad
    np.testing.assert_allclose(gamma_grad.data, expected_gamma_grad)
    x.cleargrad()
    gamma_grad.sum().backward()

    np.testing.assert_allclose(x.grad.data, np.broadcast_to(inv_std, x.shape))
