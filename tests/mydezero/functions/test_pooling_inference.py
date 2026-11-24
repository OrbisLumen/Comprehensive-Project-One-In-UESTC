"""CPU max pooling fast path and fallback behavior."""

from itertools import product

import numpy as np
import pytest

from mydezero import Variable, no_grad
from mydezero.functions import pooling
import mydezero.functions_conv as conv


def reference(x, kernel=2, stride=2, pad=0):
    patches = conv.im2col_array(x, kernel, stride, pad, to_matrix=False)
    n, c, kh, kw, oh, ow = patches.shape
    return patches.reshape(n, c, kh * kw, oh, ow).max(axis=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "interleaved", "reversed"])
def test_pooling_inference_matches_generic_on_odd_sizes(monkeypatch, dtype, layout):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 3, 9, 7)).astype(dtype)
    x[0, 1, 2, :4] = [np.nan, np.inf, -np.inf, -0.0]
    if layout == "interleaved":
        x = np.ascontiguousarray(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    elif layout == "reversed":
        x = x[:, :, ::-1, ::-1]
    x.flags.writeable = False
    expected = reference(x)

    def reject_patch_allocation(*args, **kwargs):
        pytest.fail("Eligible CPU inference should skip im2col")

    monkeypatch.setattr(conv, "im2col_array", reject_patch_allocation)
    with no_grad():
        result = pooling(x, (2, 2), stride=(2, 2), pad=(0, 0))
    assert result.creator is None
    assert result.dtype == dtype and result.data.flags.c_contiguous
    np.testing.assert_array_equal(result.data, expected)


def test_pooling_inference_preserves_signed_zero_selection():
    values = np.array(list(product((0.0, -0.0), repeat=4)), dtype=np.float32)
    x = values.reshape(1, 1, 4, 4, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(1, 1, 8, 8)
    expected = reference(x)
    with no_grad():
        actual = pooling(x, 2, stride=2).data
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("kernel,stride,pad,dtype", [
    (2, 1, 0, np.float32),
    (2, 2, 1, np.float32),
    ((2, 3), (2, 3), 0, np.float32),
    (2, 2, 0, np.int32),
])
def test_pooling_inference_uses_generic_path_for_other_inputs(monkeypatch, kernel, stride, pad, dtype):
    x = -np.arange(30, dtype=dtype).reshape(1, 1, 5, 6)
    expected = reference(x, kernel, stride, pad)
    original = conv.im2col_array
    called = []

    def spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(conv, "im2col_array", spy)
    with no_grad():
        result = pooling(x, kernel, stride, pad)
    np.testing.assert_array_equal(result.data, expected)
    assert called == [True]


def test_pooling_training_retains_indices_and_second_derivative():
    x = Variable(np.array([[[[2.0, 2.0], [1.0, 0.0]]]], dtype=np.float32))
    y = pooling(x, 2, stride=2)
    (y ** 2).sum().backward(create_graph=True)
    first = x.grad
    np.testing.assert_array_equal(first.data, [[[[4.0, 0.0], [0.0, 0.0]]]])

    x.cleargrad()
    first.sum().backward()
    np.testing.assert_array_equal(x.grad.data, [[[[2.0, 0.0], [0.0, 0.0]]]])
