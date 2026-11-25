"""Batched matrix multiplication and its gradients."""

import numpy as np
import pytest

from mydezero import Variable, no_grad
import mydezero.functions as F


def numerical_gradient(function, data, eps=1e-5):
    gradient = np.zeros_like(data)
    for index in np.ndindex(data.shape):
        above, below = data.copy(), data.copy()
        above[index] += eps
        below[index] -= eps
        with no_grad():
            gradient[index] = (function(above).data - function(below).data) / (2 * eps)
    return gradient


def test_batched_matmul_forward_matches_numpy():
    rng = np.random.default_rng(0)
    x_data = rng.standard_normal((2, 3, 4)).astype(np.float32)
    w_data = rng.standard_normal((2, 4, 5)).astype(np.float32)

    y = F.batched_matmul(x_data, w_data)

    np.testing.assert_allclose(y.data, x_data @ w_data)
    assert y.shape == (2, 3, 5)
    assert y.dtype == np.float32


def test_batched_matmul_rank_four_forward_matches_numpy():
    rng = np.random.default_rng(1)
    x_data = rng.standard_normal((2, 3, 4, 5))
    w_data = rng.standard_normal((2, 3, 5, 6))

    y = F.batched_matmul(x_data, w_data)

    np.testing.assert_allclose(y.data, x_data @ w_data)
    assert y.shape == (2, 3, 4, 6)


def test_batched_matmul_single_batch_matches_matmul():
    rng = np.random.default_rng(2)
    x_data = rng.standard_normal((3, 4))
    w_data = rng.standard_normal((4, 5))
    expected = F.matmul(x_data, w_data).data

    y = F.batched_matmul(x_data[np.newaxis], w_data[np.newaxis])

    np.testing.assert_allclose(y.data[0], expected)


def test_batched_matmul_backward_matches_numerical_gradient():
    rng = np.random.default_rng(3)
    x_data = rng.standard_normal((2, 3, 4))
    w_data = rng.standard_normal((2, 4, 5))
    x = Variable(x_data.copy())
    w = Variable(w_data.copy())

    def objective_x(values):
        return F.batched_matmul(values, w).sum()

    def objective_w(values):
        return F.batched_matmul(x, values).sum()

    objective_x(x).backward()
    np.testing.assert_allclose(x.grad.data, numerical_gradient(objective_x, x_data),
                               atol=1e-6)

    w.cleargrad()
    objective_w(w).backward()
    np.testing.assert_allclose(w.grad.data, numerical_gradient(objective_w, w_data),
                               atol=1e-6)


def test_batched_matmul_supports_higher_order_gradients():
    x = Variable(np.array([[[1.0, 2.0]]]))
    w = Variable(np.array([[[3.0], [4.0]]]))

    y = (F.batched_matmul(x, w) ** 2).sum()
    y.backward(create_graph=True)

    np.testing.assert_allclose(x.grad.data, [[[66.0, 88.0]]])
    second = x.grad
    x.cleargrad()
    second.backward()
    np.testing.assert_allclose(x.grad.data, [[[42.0, 56.0]]])


def test_batched_matmul_rejects_mismatched_operands():
    with pytest.raises(ValueError, match='same rank'):
        F.batched_matmul(np.ones((2, 3)), np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match='at least two axes'):
        F.batched_matmul(np.ones((3,)), np.ones((3,)))
    with pytest.raises(ValueError, match='leading dimensions'):
        F.batched_matmul(np.ones((2, 3, 4)), np.ones((3, 4, 5)))
