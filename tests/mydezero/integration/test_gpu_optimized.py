"""Numerical parity and higher derivatives for optimized CUDA operators."""

from itertools import product

import numpy as np
import pytest

from mydezero import Variable
import mydezero.functions as F
import mydezero.functions_conv as FC
from .test_gpu import cp


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'channels_last', 'reversed'])
def test_pointwise_forward_gradients_and_second_derivative(cp, dtype, layout):
    rng = np.random.default_rng(14)
    data = rng.normal(size=(2, 3, 5, 7)).astype(dtype)
    weight = rng.normal(size=(4, 3, 1, 1)).astype(dtype)
    host = Variable(data.copy())
    device_data = cp.asarray(data)
    if layout == 'channels_last':
        device_data = cp.ascontiguousarray(device_data.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    elif layout == 'reversed':
        device_data = device_data[..., ::-1]
        host.data = data[..., ::-1]
    device = Variable(device_data)
    weights = [Variable(weight), Variable(cp.asarray(weight))]
    a, b = [FC.conv2d(x, w) for x, w in zip((host, device), weights)]
    cp.testing.assert_allclose(b.data, cp.asarray(a.data), rtol=2e-5, atol=2e-5)
    for y in (a, b):
        (y ** 2).sum().backward(create_graph=True)
    for x, y in ((host, device), tuple(weights)):
        cp.testing.assert_allclose(y.grad.data, cp.asarray(x.grad.data), rtol=3e-5, atol=3e-5)
    ga, gb = weights[0].grad, weights[1].grad
    host.cleargrad()
    device.cleargrad()
    ga.sum().backward()
    gb.sum().backward()
    cp.testing.assert_allclose(device.grad.data, cp.asarray(host.grad.data), rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('tiles', [1, 7])
def test_pooling_edges_and_gradient_selection(cp, dtype, tiles):
    zeros = np.array(list(product((0., -0.), repeat=4)), dtype=dtype)
    patches = np.concatenate((zeros, np.array([[1, np.nan, 2, np.nan],
                                             [np.inf, 2, -np.inf, 0]], dtype=dtype)))
    x = cp.asarray(patches.reshape(-1, 1, 2, 2)).repeat(tiles, axis=3)
    y = FC.pooling(x, 2, stride=2)
    reference = FC.im2col_array(x, 2, 2, 0, to_matrix=False).reshape(-1, 1, 4, 1, tiles)
    cp.testing.assert_array_equal(y.data, reference.max(axis=2))
    cp.testing.assert_array_equal(cp.signbit(y.data), cp.signbit(reference.max(axis=2)))
    cp.testing.assert_array_equal(y.creator.indexes, reference.argmax(axis=2))


def test_pooling_noncontiguous_odd_input_and_second_derivative(cp):
    data = np.random.default_rng(5).normal(size=(2, 3, 7, 9)).astype(np.float32)
    host = Variable(data[..., ::-1])
    device = Variable(cp.asarray(data)[..., ::-1])
    for x in (host, device):
        y = FC.pooling(x, 2, stride=2)
        (y ** 2).sum().backward(create_graph=True)
    cp.testing.assert_allclose(device.grad.data, cp.asarray(host.grad.data))
    a, b = host.grad, device.grad
    host.cleargrad()
    device.cleargrad()
    a.sum().backward()
    b.sum().backward()
    cp.testing.assert_array_equal(device.grad.data, cp.asarray(host.grad.data))


@pytest.mark.parametrize('classes,batch', [(1, 1), (3, 16), (10, 64), (1000, 3)])
def test_fused_loss_and_higher_derivatives(cp, classes, batch):
    rng = np.random.default_rng(55)
    values = rng.normal(size=(batch, classes)).astype(np.float32) + np.float32(10000)
    labels = rng.integers(classes, size=(batch, 1), dtype=np.int64)
    a, b = Variable(values), Variable(cp.asarray(values))
    for create_graph in (False, True):
        a.cleargrad()
        b.cleargrad()
        la = F.softmax_cross_entropy(a, labels)
        lb = F.softmax_cross_entropy(b, cp.asarray(labels))
        (la * 0.37).backward(create_graph=create_graph)
        (lb * 0.37).backward(create_graph=create_graph)
        cp.testing.assert_allclose(lb.data, cp.asarray(la.data), rtol=2e-6, atol=2e-6)
        cp.testing.assert_allclose(b.grad.data, cp.asarray(a.grad.data), rtol=3e-5, atol=2e-6)
        if create_graph:
            ga, gb = a.grad, b.grad
            a.cleargrad()
            b.cleargrad()
            (ga ** 2).sum().backward()
            (gb ** 2).sum().backward()
            cp.testing.assert_allclose(b.grad.data, cp.asarray(a.grad.data), rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_loss_extremes_and_nondefault_stream(cp, dtype):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x = Variable(cp.array([[1000, -1000, 0], [-1000, 1000, 0]], dtype=dtype))
        loss = F.softmax_cross_entropy(x, cp.array([1, 1], dtype=cp.int32))
        loss.backward()
    stream.synchronize()
    assert loss.dtype == dtype and x.grad.dtype == dtype
    cp.testing.assert_allclose(loss.data, cp.asarray(1000, dtype=dtype))
    cp.testing.assert_allclose(x.grad.data, cp.array([[0.5, -0.5, 0], [0, 0, 0]], dtype=dtype))


@pytest.mark.parametrize('create_graph', [False, True])
def test_batchnorm_and_global_pooling_preserve_fp32_gradients(cp, create_graph):
    from mydezero.models import _global_average_pooling_2d
    random = np.random.default_rng(21)
    x = Variable(cp.asarray(random.normal(size=(2, 3, 5, 7)).astype(np.float32)))
    gamma = Variable(cp.ones(3, dtype=cp.float32))
    beta = Variable(cp.zeros(3, dtype=cp.float32))
    y = F.batch_norm(x, gamma, beta, cp.zeros(3, dtype=cp.float32), cp.ones(3, dtype=cp.float32))
    pooled = _global_average_pooling_2d(y)
    assert pooled.dtype == cp.float32
    F.softmax_cross_entropy(pooled, cp.array([0, 2])).backward(create_graph=create_graph)
    for parameter in (x, gamma, beta):
        assert parameter.grad.dtype == cp.float32


@pytest.mark.parametrize('gradient_order', ['C', 'F'])
def test_fused_adam_updates_fortran_weights_and_moments(cp, gradient_order):
    from mydezero import Parameter
    from mydezero.optimizers import Adam
    random = np.random.default_rng(82)
    pairs = []
    for fused in (False, True):
        weight = Parameter(cp.array(np.ones((32, 17), dtype=np.float32), order='F'))
        pairs.append((weight, Adam(lr=0.003, fused=fused)))
    for _ in range(8):
        values = random.normal(size=(32, 17)).astype(np.float32)
        for weight, optimizer in pairs:
            weight.grad = Variable(cp.array(values, order=gradient_order))
            optimizer.update_one(weight)
    (a, reference), (b, optimized) = pairs
    for expected, actual in ((a.data, b.data), (reference.ms[id(a)], optimized.ms[id(b)]),
                             (reference.vs[id(a)], optimized.vs[id(b)])):
        assert actual.flags.f_contiguous
        cp.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
