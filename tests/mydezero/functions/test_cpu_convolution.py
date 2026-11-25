import numpy as np
import pytest

from mydezero import Variable
from mydezero import functions_conv as conv


@pytest.mark.parametrize(
    'kernel, stride, pad, layout, dtype, with_bias',
    [
        ((3, 2), (2, 1), (1, 2), 'nchw', np.float64, True),
        ((3, 3), (1, 1), (0, 0), 'nhwc', np.float32, False),
        ((2, 3), (2, 3), (0, 0), 'strided', np.float32, True),
        ((1, 1), (1, 1), (0, 0), 'nchw', np.float32, True),
        ((1, 1), (1, 1), (0, 0), 'nhwc', np.float64, False),
        ((1, 1), (1, 1), (0, 0), 'strided', np.float32, False),
        ((1, 1), (2, 1), (1, 0), 'nchw', np.float32, True),
    ],
)
def test_cpu_convolution_matches_reference(kernel, stride, pad, layout, dtype, with_bias):
    rng = np.random.default_rng(9)
    x_data = rng.standard_normal((2, 3, 7, 9)).astype(dtype)
    if layout == 'nhwc':
        x_data = np.ascontiguousarray(x_data.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    if layout == 'strided':
        x_data = x_data[..., ::-1, ::2]

    weight = rng.standard_normal((5, 3, *kernel)).astype(dtype)[..., ::-1]
    bias = rng.standard_normal(5).astype(dtype) if with_bias else None
    columns = conv.im2col_array(x_data, kernel, stride, pad, to_matrix=False)
    expected = np.tensordot(columns, weight, ((1, 2, 3), (1, 2, 3)))
    if with_bias:
        expected += bias
    expected = np.moveaxis(expected, -1, 1)
    gy = rng.standard_normal(expected.shape).astype(dtype)[..., ::-1]

    x, w, b = Variable(x_data), Variable(weight), Variable(bias)
    actual = conv.conv2d(x, w, b, stride=stride, pad=pad)
    actual.grad = Variable(gy)
    actual.backward(create_graph=True)

    atol = 3e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual.data, expected, rtol=0, atol=atol)
    np.testing.assert_allclose(x.grad.data, conv.deconv2d(
        gy, weight, stride=stride, pad=pad, outsize=x_data.shape[2:]
    ).data, rtol=0, atol=atol)
    np.testing.assert_allclose(w.grad.data, np.tensordot(
        gy, columns, ((0, 2, 3), (0, 4, 5))
    ), rtol=0, atol=atol)
    if with_bias:
        np.testing.assert_allclose(b.grad.data, gy.sum(axis=(0, 2, 3)), rtol=0, atol=atol)

    grad_weight = w.grad
    x.cleargrad()
    grad_weight.backward()
    np.testing.assert_allclose(x.grad.data, conv.deconv2d(
        gy, np.ones_like(weight), stride=stride, pad=pad, outsize=x_data.shape[2:]
    ).data, rtol=0, atol=atol)


def test_cpu_convolution_preserves_mixed_and_integer_dtypes():
    for dtype, weight_dtype in ((np.float32, np.float64), (np.int32, np.int32)):
        x = np.arange(12, dtype=dtype).reshape(1, 2, 2, 3)
        w = np.array([1, 2], dtype=weight_dtype).reshape(1, 2, 1, 1)
        actual = conv.conv2d(x, w).data
        expected = (x[:, :1] + 2 * x[:, 1:]).astype(np.result_type(x, w))
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
