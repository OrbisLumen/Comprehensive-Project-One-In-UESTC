"""Optional FP32 cuDNN convolution probe used by the GPU experiment."""

import ctypes as ct
from ctypes.util import find_library
import os
from pathlib import Path

import cupy as cp


def load_library(torch):
    """Resolve an explicit library or the library shipped with benchmark PyTorch."""
    configured = os.environ.get('MYDEZERO_CUDNN_LIBRARY')
    directory = Path(torch.__file__).parent / 'lib'
    if os.name == 'nt':
        name = configured or str(directory / 'cudnn64_9.dll')
        # cuDNN's split libraries are loaded dynamically through the search path.
        os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')
    else:
        name = configured or find_library('cudnn')
    if not name:
        raise OSError('Set MYDEZERO_CUDNN_LIBRARY to an installed cuDNN library.')
    return ct.CDLL(name)


class Convolution:
    """Reuse descriptors for one contiguous FP32 convolution fixture on one device."""

    def __init__(self, library, x, weight, pad=0):
        self.library = library
        self.device = cp.cuda.runtime.getDevice()
        self.resources = []
        self.alpha, self.beta = ct.c_float(1), ct.c_float(0)
        self.x_shape, self.w_shape = x.shape, weight.shape
        n, c, h, w = x.shape
        oc, _, kh, kw = weight.shape
        self.y_shape = (n, oc, h + 2 * pad - kh + 1, w + 2 * pad - kw + 1)
        try:
            self.handle = self.create('cudnnCreate', 'cudnnDestroy')
            self.x_desc = self.tensor(self.x_shape)
            self.y_desc = self.tensor(self.y_shape)
            self.w_desc = self.create('cudnnCreateFilterDescriptor', 'cudnnDestroyFilterDescriptor')
            self.call('cudnnSetFilter4dDescriptor', self.w_desc, 0, 0, *weight.shape)
            self.conv = self.create('cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
            self.call('cudnnSetConvolution2dDescriptor', self.conv, pad, pad, 1, 1, 1, 1, 1, 0)
            # FMA math requests FP32 without TF32 conversion.
            self.call('cudnnSetConvolutionMathType', self.conv, 2)
        except Exception:
            self.close()
            raise

    def call(self, name, *args):
        function = getattr(self.library, name)
        function.restype = ct.c_int
        status = function(*args)
        if status:
            raise RuntimeError(f'{name} returned cuDNN status {status}')

    def create(self, create, destroy):
        handle = ct.c_void_p()
        self.call(create, ct.byref(handle))
        self.resources.append((destroy, handle))
        return handle

    def tensor(self, shape):
        descriptor = self.create('cudnnCreateTensorDescriptor', 'cudnnDestroyTensorDescriptor')
        self.call('cudnnSetTensor4dDescriptor', descriptor, 0, 0, *shape)
        return descriptor

    def run(self, direction, x, weight, gradient=None):
        if cp.cuda.runtime.getDevice() != self.device:
            raise ValueError('The probe must run on its original device.')
        arrays = (x, weight) if gradient is None else (x, weight, gradient)
        if any(a.dtype != cp.float32 or not a.flags.c_contiguous for a in arrays):
            raise ValueError('The probe requires contiguous FP32 device arrays.')
        if x.shape != self.x_shape or weight.shape != self.w_shape:
            raise ValueError('Create a new probe for a different shape.')
        if gradient is not None and gradient.shape != self.y_shape:
            raise ValueError('Upstream gradient shape differs from convolution output.')
        self.call('cudnnSetStream', self.handle, ct.c_void_p(cp.cuda.get_current_stream().ptr))
        alpha, beta = ct.byref(self.alpha), ct.byref(self.beta)
        ptr = lambda a: ct.c_void_p(a.data.ptr)
        if direction == 'Forward':
            out = cp.empty(self.y_shape, dtype=cp.float32)
            self.call('cudnnConvolutionForward', self.handle, alpha, self.x_desc, ptr(x),
                      self.w_desc, ptr(weight), self.conv, 0, None, ct.c_size_t(0),
                      beta, self.y_desc, ptr(out))
        elif direction == 'dX':
            out = cp.empty_like(x)
            self.call('cudnnConvolutionBackwardData', self.handle, alpha, self.w_desc, ptr(weight),
                      self.y_desc, ptr(gradient), self.conv, 0, None, ct.c_size_t(0),
                      beta, self.x_desc, ptr(out))
        elif direction == 'dW':
            out = cp.empty_like(weight)
            self.call('cudnnConvolutionBackwardFilter', self.handle, alpha, self.x_desc, ptr(x),
                      self.y_desc, ptr(gradient), self.conv, 0, None, ct.c_size_t(0),
                      beta, self.w_desc, ptr(out))
        else:
            raise ValueError(direction)
        return out

    def close(self):
        for name, descriptor in reversed(self.resources):
            self.call(name, descriptor)
        self.resources.clear()
