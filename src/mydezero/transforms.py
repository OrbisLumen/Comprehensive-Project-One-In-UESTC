"""Composable transforms for individual NumPy samples."""

import numpy as np


def _positive_int(value, name):
    """Return value as an int, rejecting non-integer and boolean inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _non_negative_int(value, name):
    """Return value as an int, rejecting non-integer and boolean inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return int(value)


def _spatial_shape(array, name):
    """Return validated trailing spatial dimensions of a (..., H, W) array."""
    if array.ndim < 2:
        raise ValueError(f'{name} expects an array with at least two dimensions')
    height, width = array.shape[-2:]
    if height == 0 or width == 0:
        raise ValueError(f'{name} expects nonempty spatial dimensions')
    return height, width


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list): list of transforms
    """

    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================
class CentralCrop:
    """Crop the trailing spatial axes around the center.

    Args:
        size (int): Side length of the square crop.

    Raises:
        ValueError: If size is not a positive integer, or if the input has fewer
            than two dimensions, empty spatial axes, or dimensions smaller than
            size when called.
    """

    def __init__(self, size):
        self.size = _positive_int(size, 'CentralCrop size')

    def __call__(self, array):
        height, width = _spatial_shape(array, 'CentralCrop')
        if self.size > min(height, width):
            raise ValueError('CentralCrop size must not exceed the input dimensions')
        top = (height - self.size) // 2
        left = (width - self.size) // 2
        return array[..., top:top + self.size, left:left + self.size].copy()


class RandomCrop:
    """Crop the trailing spatial axes at a random offset.

    Constant padding is applied to both spatial borders before cropping when
    padding is positive. The fill value is cast to the input dtype by NumPy.

    Args:
        size (int): Side length of the square crop.
        padding (int): Constant padding added to every spatial border first.
        fill (float): Constant value used for padding, cast to the input dtype.
        seed (int or None): Seed for the random generator. Reuse the same seed to
            reproduce the sampled crop positions.

    Raises:
        ValueError: If size or padding is invalid, or if the input has fewer than
            two dimensions or the crop exceeds the padded spatial dimensions when
            called.
    """

    def __init__(self, size, padding=0, fill=0, seed=None):
        self.size = _positive_int(size, 'RandomCrop size')
        self.padding = _non_negative_int(padding, 'RandomCrop padding')
        self.fill = fill
        self.rng = np.random.default_rng(seed)

    def __call__(self, array):
        if array.ndim < 2:
            raise ValueError('RandomCrop expects an array with at least two dimensions')
        if self.padding:
            pad = [(0, 0)] * (array.ndim - 2) + [(self.padding, self.padding)] * 2
            array = np.pad(array, pad, mode='constant', constant_values=self.fill)
        height, width = _spatial_shape(array, 'RandomCrop')
        if self.size > min(height, width):
            raise ValueError('RandomCrop size must not exceed the padded dimensions')
        top = int(self.rng.integers(0, height - self.size + 1))
        left = int(self.rng.integers(0, width - self.size + 1))
        return array[..., top:top + self.size, left:left + self.size].copy()


class RandomHorizontalFlip:
    """Flip the trailing width axis with the given probability.

    Args:
        p (float): Probability of flipping. 0 never flips and 1 always flips.
        seed (int or None): Seed for the random generator.

    Raises:
        ValueError: If p is outside the closed interval [0, 1], or if the input
            has fewer than one dimension when called.
    """

    def __init__(self, p=0.5, seed=None):
        if (isinstance(p, bool)
                or not isinstance(p, (int, float, np.integer, np.floating))
                or not 0.0 <= p <= 1.0):
            raise ValueError('RandomHorizontalFlip p must be a number between 0 and 1')
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, array):
        if array.ndim < 1:
            raise ValueError('RandomHorizontalFlip expects an array with at least one dimension')
        if self.rng.random() < self.p:
            return array[..., ::-1].copy()
        return array.copy()


def _resample_axis(array, axis, out_size):
    """Linearly resample one axis with half-pixel centers."""
    axis %= array.ndim
    in_size = array.shape[axis]
    if in_size == out_size:
        return array
    coord = (np.arange(out_size) + 0.5) * (in_size / out_size) - 0.5
    coord = np.clip(coord, 0.0, in_size - 1)
    lower = np.floor(coord).astype(np.intp)
    upper = np.minimum(lower + 1, in_size - 1)
    shape = [1] * array.ndim
    shape[axis] = out_size
    weight = (coord - lower).reshape(shape)
    low = np.take(array, lower, axis=axis)
    high = np.take(array, upper, axis=axis)
    return (1.0 - weight) * low + weight * high


class Resize:
    """Resize the trailing spatial axes with bilinear interpolation.

    Args:
        size (int or tuple[int, int]): Square side length, or a (height, width)
            pair of positive integers.

    Raises:
        ValueError: If size is not a positive integer or a pair of positive
            integers, or if the input has fewer than two dimensions or empty
            spatial axes when called.
        TypeError: If the input dtype is not a real numeric or boolean type.
    """

    def __init__(self, size):
        if isinstance(size, (int, np.integer)) and not isinstance(size, bool):
            self.size = (_positive_int(size, 'Resize size'),) * 2
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            self.size = (_positive_int(size[0], 'Resize size'),
                         _positive_int(size[1], 'Resize size'))
        else:
            raise ValueError('Resize size must be a positive integer or a (height, width) pair')

    def __call__(self, array):
        _spatial_shape(array, 'Resize')
        dtype = array.dtype
        if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)
                or dtype == np.bool_):
            raise TypeError('Resize expects a real numeric array')
        result = _resample_axis(np.asarray(array, dtype=np.float64), -2, self.size[0])
        result = _resample_axis(result, -1, self.size[1])
        if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
            return np.rint(result).astype(dtype)
        return result.astype(dtype)


class Normalize:
    """Compute (array - mean) / std using nonzero standard deviations.

    Channel sequences broadcast along the first axis (channel-first data).
    Convert integer pixels to floating point before normalization.

    Args:
        mean (float or sequence): mean for all values or sequence of means for
         each channel.
        std (float or sequence): standard deviation for each channel.
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """Flatten a NumPy array.
    """

    def __call__(self, array):
        return array.flatten()


class AsType:
    """Return an array converted to dtype (float32 by default).

    ToFloat is an alias for this class and accepts the same dtype argument.
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    """Convert an array to an integer dtype (platform int by default)."""

    def __init__(self, dtype=int):
        self.dtype = dtype
