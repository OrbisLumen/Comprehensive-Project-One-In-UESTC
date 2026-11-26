"""Optional cuDNN convolution gradients, descriptor reuse and stream behavior."""

import numpy as np
import pytest

from mydezero import Variable, using_config
import mydezero.functions_conv as FC
from .test_gpu import cp


@pytest.fixture
def cudnn(cp):
    from mydezero import _cudnn
    try:
        _cudnn.library()
    except RuntimeError as error:
        pytest.skip(str(error))
    return _cudnn


@pytest.mark.parametrize('kernel,stride,pad', [((1, 1), 1, 0), ((3, 2), (2, 1), (1, 2))])
@pytest.mark.parametrize('operation', [FC.conv2d, FC.deconv2d])
def test_cudnn_forward_and_gradients(cp, cudnn, kernel, stride, pad, operation):
    random = np.random.default_rng(17)
    wshape = (4, 3, *kernel) if operation is FC.conv2d else (3, 4, *kernel)
    arrays = [random.normal(size=(2, 3, 7, 9)).astype(np.float32),
              random.normal(size=wshape).astype(np.float32), random.normal(size=4).astype(np.float32)]
    cpu = [Variable(a.copy()) for a in arrays]
    gpu = [Variable(cp.asarray(a)) for a in arrays]
    expected = operation(*cpu, stride=stride, pad=pad)
    with using_config('use_cudnn', True):
        actual = operation(*gpu, stride=stride, pad=pad)
        gradient = random.normal(size=actual.shape).astype(np.float32)
        expected.grad, actual.grad = Variable(gradient), Variable(cp.asarray(gradient))
        expected.backward(create_graph=True)
        actual.backward(create_graph=True)
        cp.testing.assert_allclose(actual.data, cp.asarray(expected.data), rtol=4e-4, atol=4e-4)
        for host, device in zip(cpu, gpu):
            cp.testing.assert_allclose(device.grad.data, cp.asarray(host.grad.data), rtol=5e-4, atol=5e-4)
        first = gpu[1].grad
        expected_first = cpu[1].grad
        gpu[0].cleargrad()
        cpu[0].cleargrad()
        first.sum().backward()
        expected_first.sum().backward()
        cp.testing.assert_allclose(gpu[0].grad.data, cp.asarray(cpu[0].grad.data), rtol=5e-4, atol=5e-4)


def test_cudnn_refreshes_weight_values_and_supports_nondefault_stream(cp, cudnn):
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, using_config('use_cudnn', True):
        x = cp.ones((2, 3, 8, 8), dtype=cp.float32)[..., ::-1]
        w = cp.ones((4, 3, 3, 3), dtype=cp.float32)
        a = FC.conv2d(x, w).data
        w *= 2
        b = FC.conv2d(x, w).data
    stream.synchronize()
    cp.testing.assert_allclose(b, a * 2)


def test_cudnn_keeps_fp64_on_generic_path(cp, cudnn, monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail('FP64 must use the existing path')
    monkeypatch.setattr(cudnn, 'execute', reject)
    with using_config('use_cudnn', True):
        out = FC.conv2d(cp.ones((1, 2, 4, 4), dtype=cp.float64),
                        cp.ones((3, 2, 2, 2), dtype=cp.float64))
    cp.testing.assert_array_equal(out.data, cp.full((1, 3, 3, 3), 8, dtype=cp.float64))
