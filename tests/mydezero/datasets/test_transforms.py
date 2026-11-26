"""Transform order, channel broadcasting, dtype, and input preservation."""

import numpy as np
import pytest

from mydezero.transforms import (AsType, CentralCrop, Compose, Flatten, Normalize,
                                 RandomCrop, RandomHorizontalFlip, Resize, ToFloat,
                                 ToInt)


def test_compose_identity_and_order():
    x = np.array([1, 2])
    assert Compose()(x) is x
    np.testing.assert_array_equal(Compose([lambda a: a + 1, lambda a: a * 3])(x), [6, 9])


def test_mnist_transform_preserves_raw_pixels():
    x = np.array([[[0, 255], [127, 64]]], dtype=np.uint8)
    original = x.copy()
    result = Compose([Flatten(), ToFloat(), Normalize(0, 255)])(x)
    assert result.shape == (4,) and result.dtype == np.float32
    np.testing.assert_allclose(result, [0, 1, 127 / 255, 64 / 255])
    np.testing.assert_array_equal(x, original)


def test_normalize_channel_first_values():
    x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    result = Normalize([1, 4], [2, 3])(x)
    np.testing.assert_allclose(result[0], (x[0] - 1) / 2)
    np.testing.assert_allclose(result[1], (x[1] - 4) / 3)
    assert result.dtype == np.float32


@pytest.mark.parametrize('transform,dtype', [
    (AsType(), np.float32), (ToFloat(np.float64), np.float64),
    (ToInt(), np.dtype(int)), (ToInt(np.int32), np.int32),
])
def test_dtype_transforms(transform, dtype):
    x = np.array([1.0, 2.0])
    result = transform(x)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, x)


def test_central_crop_uses_centered_window():
    x = np.arange(1 * 5 * 6, dtype=np.uint8).reshape(1, 5, 6)

    result = CentralCrop(3)(x)

    assert result.shape == (1, 3, 3)
    np.testing.assert_array_equal(result, x[:, 1:4, 1:4])
    assert result.dtype == np.uint8


def test_central_crop_accepts_two_dimensional_input():
    x = np.arange(25).reshape(5, 5)

    np.testing.assert_array_equal(CentralCrop(3)(x), x[1:4, 1:4])


@pytest.mark.parametrize('size', [0, -1, 2.5])
def test_central_crop_rejects_invalid_size(size):
    with pytest.raises(ValueError, match='positive integer'):
        CentralCrop(size)


def test_central_crop_rejects_oversized_crop():
    with pytest.raises(ValueError, match='exceed'):
        CentralCrop(7)(np.zeros((1, 5, 6)))


def test_random_crop_is_valid_and_seeded_reproducible():
    x = np.arange(2 * 6 * 7).reshape(2, 6, 7)
    first = RandomCrop(4, seed=0)(x)
    second = RandomCrop(4, seed=0)(x)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 4, 4)
    crops = [x[..., top:top + 4, left:left + 4]
             for top in range(3) for left in range(4)]
    assert any(np.array_equal(first, crop) for crop in crops)
    assert RandomCrop(4, seed=1)(x).shape == (2, 4, 4)


def test_random_crop_pads_before_cropping():
    x = np.ones((1, 4, 4), dtype=np.uint8)

    result = RandomCrop(8, padding=2, seed=0)(x)

    assert result.shape == (1, 8, 8)
    assert result.min() == 0 and result.max() == 1


def test_random_crop_uses_custom_padding_fill_and_dtype():
    x = np.ones((1, 4, 4), dtype=np.uint8)

    result = RandomCrop(8, padding=2, fill=7, seed=0)(x)

    assert result.shape == (1, 8, 8) and result.dtype == np.uint8
    assert result[0, 0, 0] == 7
    assert result[0, 2, 2] == 1


def test_random_crop_rejects_oversized_crop_after_padding():
    with pytest.raises(ValueError, match='padded'):
        RandomCrop(9, padding=2, seed=0)(np.zeros((1, 4, 4)))


def test_random_crop_preserves_input_and_dtype():
    x = np.arange(1 * 5 * 5, dtype=np.uint8).reshape(1, 5, 5)
    original = x.copy()

    result = RandomCrop(3, seed=2)(x)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize('p, flipped', [(0.0, False), (1.0, True)])
def test_random_horizontal_flip_probability_extremes(p, flipped):
    x = np.arange(1 * 3 * 4).reshape(1, 3, 4)

    result = RandomHorizontalFlip(p=p, seed=0)(x)

    expected = x[..., ::-1] if flipped else x
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == x.dtype


def test_random_horizontal_flip_seed_is_reproducible():
    x = np.arange(10).reshape(2, 5)

    first = RandomHorizontalFlip(seed=3)(x)
    second = RandomHorizontalFlip(seed=3)(x)

    np.testing.assert_array_equal(first, second)


def test_random_horizontal_flip_rejects_invalid_probability():
    with pytest.raises(ValueError, match='between 0 and 1'):
        RandomHorizontalFlip(p=1.5)


@pytest.mark.parametrize('p', ['0.5', None, True])
def test_random_horizontal_flip_rejects_non_numeric_probability(p):
    with pytest.raises(ValueError, match='between 0 and 1'):
        RandomHorizontalFlip(p=p)


def test_crop_and_flip_do_not_mutate_input():
    x = np.arange(16).reshape(1, 4, 4)
    original = x.copy()

    CentralCrop(2)(x)
    RandomCrop(2, seed=0)(x)
    RandomHorizontalFlip(1.0, seed=0)(x)

    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize('transform', [CentralCrop(1), RandomCrop(1, seed=0), Resize(1)])
def test_spatial_transforms_reject_fewer_than_two_dimensions(transform):
    for value in (np.array(1.0), np.arange(4)):
        with pytest.raises(ValueError, match='at least two dimensions'):
            transform(value)


def test_resize_rejects_empty_spatial_dimensions():
    with pytest.raises(ValueError, match='nonempty spatial dimensions'):
        Resize(2)(np.zeros((2, 0, 5)))


@pytest.mark.parametrize('array', [np.array([[1 + 2j]]), np.array([[object()]])])
def test_resize_rejects_non_real_numeric_dtypes(array):
    with pytest.raises(TypeError, match='real numeric'):
        Resize(2)(array)


@pytest.mark.parametrize('factory', [lambda: CentralCrop(True), lambda: RandomCrop(True),
                                     lambda: RandomCrop(1, padding=True),
                                     lambda: Resize((True, 2))])
def test_spatial_transforms_reject_boolean_integer_parameters(factory):
    with pytest.raises(ValueError, match='positive integer|non-negative integer'):
        factory()


def test_random_horizontal_flip_rejects_zero_dimensional_input():
    with pytest.raises(ValueError, match='at least one dimension'):
        RandomHorizontalFlip()(np.array(1.0))


def test_resize_bilinear_known_values():
    x = np.array([[0.0, 10.0], [20.0, 30.0]])

    result = Resize(3)(x)

    expected = np.array([[0.0, 5.0, 10.0],
                         [10.0, 15.0, 20.0],
                         [20.0, 25.0, 30.0]])
    np.testing.assert_allclose(result, expected)
    assert result.shape == (3, 3)


def test_resize_channel_and_rectangle_shapes():
    x = np.arange(1 * 4 * 6, dtype=np.uint8).reshape(1, 4, 6)

    result = Resize((2, 3))(x)

    assert result.shape == (1, 2, 3)
    assert result.dtype == np.uint8


def test_resize_preserves_integer_dtype_and_float_dtype():
    integer = np.array([[0, 200]], dtype=np.uint8)

    resized = Resize((1, 3))(integer)

    np.testing.assert_array_equal(resized, np.array([[0, 100, 200]], dtype=np.uint8))
    assert resized.dtype == np.uint8
    assert Resize(2)(np.ones((2, 2), dtype=np.float32)).dtype == np.float32


def test_resize_identity_returns_equal_array():
    x = np.arange(9, dtype=np.float32).reshape(3, 3)

    np.testing.assert_array_equal(Resize(3)(x), x)


@pytest.mark.parametrize('size', [0, -2, (2, 0), (2.5, 3), 2.5, (2, 3, 4)])
def test_resize_rejects_invalid_size(size):
    with pytest.raises(ValueError, match='positive'):
        Resize(size)


def test_resize_does_not_mutate_input():
    x = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
    original = x.copy()

    Resize((2, 2))(x)

    np.testing.assert_array_equal(x, original)
