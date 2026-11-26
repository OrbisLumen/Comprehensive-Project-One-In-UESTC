"""Optional FP32 CUDA convolution using a separately installed cuDNN library."""

from collections import OrderedDict
import ctypes as ct
from ctypes.util import find_library
from functools import lru_cache
import os
from pathlib import Path
import threading
import weakref

from mydezero import cuda


_state = threading.local()
_dll_directories = []
_FMA_MATH = 3


class _Performance(ct.Structure):
    _fields_ = [('algorithm', ct.c_int), ('status', ct.c_int), ('time', ct.c_float),
                ('memory', ct.c_size_t), ('determinism', ct.c_int), ('math_type', ct.c_int),
                ('reserved', ct.c_int * 3)]


@lru_cache(maxsize=1)
def library():
    path = os.environ.get('MYDEZERO_CUDNN_LIBRARY') or find_library('cudnn')
    if not path:
        path = 'cudnn64_9.dll' if os.name == 'nt' else 'libcudnn.so.9'
    if os.name == 'nt' and Path(path).is_absolute():
        directory = str(Path(path).parent)
        _dll_directories.append(os.add_dll_directory(directory))
        os.environ['PATH'] = directory + os.pathsep + os.environ.get('PATH', '')
    try:
        return ct.CDLL(path)
    except OSError as error:
        raise RuntimeError('Install cuDNN or set MYDEZERO_CUDNN_LIBRARY to its shared library.') from error


def _release(shared_library, resources):
    for name, descriptor in reversed(resources):
        getattr(shared_library, name)(descriptor)
    resources.clear()


class Convolution:
    """Descriptors for one device, shape, stride and padding configuration."""

    def __init__(self, shared_library, x_shape, w_shape, stride=(1, 1), pad=(0, 0)):
        self.library = shared_library
        self.resources = []
        self._release = weakref.finalize(self, _release, shared_library, self.resources)
        self.alpha, self.beta = ct.c_float(1), ct.c_float(0)
        n, _, height, width = x_shape
        oc, _, kh, kw = w_shape
        self.y_shape = (n, oc, (height + 2 * pad[0] - kh) // stride[0] + 1,
                        (width + 2 * pad[1] - kw) // stride[1] + 1)
        self.x_shape, self.w_shape = x_shape, w_shape
        self.handle = self.create('cudnnCreate', 'cudnnDestroy')
        self.x_desc = self.tensor(x_shape)
        self.y_desc = self.tensor(self.y_shape)
        self.w_desc = self.create('cudnnCreateFilterDescriptor', 'cudnnDestroyFilterDescriptor')
        self.call('cudnnSetFilter4dDescriptor', self.w_desc, 0, 0, *w_shape)
        self.conv = self.create('cudnnCreateConvolutionDescriptor', 'cudnnDestroyConvolutionDescriptor')
        self.call('cudnnSetConvolution2dDescriptor', self.conv, *pad, *stride, 1, 1, 1, 0)
        self.call('cudnnSetConvolutionMathType', self.conv, _FMA_MATH)
        self.algorithms = {}

    def algorithm(self, direction):
        if direction not in self.algorithms:
            count = ct.c_int()
            performance = (_Performance * 8)()
            descriptors = {
                'forward': (self.x_desc, self.w_desc, self.conv, self.y_desc),
                'data': (self.w_desc, self.y_desc, self.conv, self.x_desc),
                'weight': (self.x_desc, self.y_desc, self.conv, self.w_desc),
            }
            function = {'forward': 'cudnnGetConvolutionForwardAlgorithm_v7',
                        'data': 'cudnnGetConvolutionBackwardDataAlgorithm_v7',
                        'weight': 'cudnnGetConvolutionBackwardFilterAlgorithm_v7'}[direction]
            self.call(function, self.handle, *descriptors[direction], 8, ct.byref(count), performance)
            # Use direct/GEMM algorithms for the initial strict-FP32 backend.
            allowed = (0, 1, 2, 3) if direction == 'forward' else (0, 1)
            candidates = [p for p in performance[:count.value]
                          if p.status == 0 and p.algorithm in allowed
                          and p.math_type in (0, _FMA_MATH) and p.memory <= 256 * 1024 * 1024]
            if not candidates:
                raise RuntimeError('No cuDNN convolution algorithm fits the 256 MiB workspace limit.')
            selected = candidates[0]
            self.algorithms[direction] = selected.algorithm, selected.memory
        return self.algorithms[direction]

    def call(self, name, *args):
        status = getattr(self.library, name)(*args)
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
        cp = cuda.cupy
        self.call('cudnnSetStream', self.handle, ct.c_void_p(cp.cuda.get_current_stream().ptr))
        algorithm, workspace_bytes = self.algorithm(direction)
        workspace = cp.empty(workspace_bytes, dtype=cp.uint8)
        workspace_ptr = ct.c_void_p(workspace.data.ptr) if workspace_bytes else None
        alpha, beta = ct.byref(self.alpha), ct.byref(self.beta)
        weight_ptr = None if weight is None else ct.c_void_p(weight.data.ptr)
        if direction == 'forward':
            out = cp.empty(self.y_shape, dtype=cp.float32)
            self.call('cudnnConvolutionForward', self.handle, alpha, self.x_desc, ct.c_void_p(x.data.ptr),
                      self.w_desc, weight_ptr, self.conv, algorithm, workspace_ptr, ct.c_size_t(workspace_bytes),
                      beta, self.y_desc, ct.c_void_p(out.data.ptr))
        elif direction == 'data':
            out = cp.empty(self.x_shape, dtype=cp.float32)
            self.call('cudnnConvolutionBackwardData', self.handle, alpha, self.w_desc, weight_ptr,
                      self.y_desc, ct.c_void_p(gradient.data.ptr), self.conv, algorithm, workspace_ptr, ct.c_size_t(workspace_bytes),
                      beta, self.x_desc, ct.c_void_p(out.data.ptr))
        elif direction == 'weight':
            out = cp.empty(self.w_shape, dtype=cp.float32)
            self.call('cudnnConvolutionBackwardFilter', self.handle, alpha, self.x_desc, ct.c_void_p(x.data.ptr),
                      self.y_desc, ct.c_void_p(gradient.data.ptr), self.conv, algorithm, workspace_ptr, ct.c_size_t(workspace_bytes),
                      beta, self.w_desc, ct.c_void_p(out.data.ptr))
        else:
            raise ValueError(direction)
        return out


def eligible(*arrays):
    if not cuda.gpu_enable:
        return False
    cp = cuda.cupy
    if not all(isinstance(a, cp.ndarray) and a.dtype == cp.float32 and a.ndim == 4
               and min(a.shape) > 0 for a in arrays):
        return False
    device = cp.cuda.runtime.getDevice()
    return all(a.device.id == device for a in arrays)


def execute(direction, x, weight, stride, pad, gradient=None, input_shape=None, weight_shape=None):
    cp = cuda.cupy
    x_shape = x.shape if input_shape is None else tuple(input_shape)
    cache = getattr(_state, 'primitives', None)
    if cache is None:
        cache = _state.primitives = OrderedDict()
    w_shape = weight.shape if weight_shape is None else tuple(weight_shape)
    key = (cp.cuda.runtime.getDevice(), x_shape, w_shape, tuple(stride), tuple(pad))
    if key not in cache:
        cache[key] = Convolution(library(), x_shape, w_shape, stride, pad)
        if len(cache) > 64:
            cache.popitem(last=False)
    cache.move_to_end(key)
    descriptor = cache[key]
    if gradient is not None and gradient.shape != descriptor.y_shape:
        raise ValueError('Upstream gradient shape does not match the convolution descriptor.')
    return descriptor.run(direction, cp.ascontiguousarray(x),
                          None if weight is None else cp.ascontiguousarray(weight),
                          None if gradient is None else cp.ascontiguousarray(gradient))
