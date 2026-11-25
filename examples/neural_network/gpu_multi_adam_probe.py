"""Restricted multi-tensor Adam prototype for equal-size FP32 benchmark tensors."""

import cupy as cp
import numpy as np


kernel = cp.RawKernel(r'''
extern "C" __global__ void multi_adam(const unsigned long long* pointers,
    int size, float correction1, float correction2, float lr, float epsilon) {
    int tensor = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float* weight = (float*)pointers[tensor * 4];
    const float* grad = (const float*)pointers[tensor * 4 + 1];
    float* m = (float*)pointers[tensor * 4 + 2];
    float* v = (float*)pointers[tensor * 4 + 3];
    float first = 0.9f * m[i] + 0.1f * grad[i];
    float second = 0.999f * v[i] + 0.001f * powf(grad[i], 2.0f);
    m[i] = first;
    v[i] = second;
    weight[i] -= lr * ((first / correction1) / (sqrtf(second / correction2) + epsilon));
}
''', 'multi_adam', options=('--fmad=false',))


class EqualSizeAdam:
    """Benchmark fixed equal-size, independent contiguous FP32 parameter lists."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.size = parameters[0].size
        if any(p.data.dtype != cp.float32 or not p.data.flags.c_contiguous
               or p.size != self.size for p in parameters):
            raise ValueError('The prototype supports equal-size contiguous FP32 arrays.')
        self.ms = [cp.zeros_like(p.data) for p in parameters]
        self.vs = [cp.zeros_like(p.data) for p in parameters]
        self.step = 0

    def update(self):
        if any(p.grad is None for p in self.parameters):
            raise ValueError('Every prototype parameter must have a gradient.')
        # Include pointer-list construction and transfer in the timed operation.
        pointers = np.array([[p.data.data.ptr, p.grad.data.data.ptr, m.data.ptr, v.data.ptr]
                             for p, m, v in zip(self.parameters, self.ms, self.vs)], dtype=np.uint64)
        device_pointers = cp.asarray(pointers)
        self.step += 1
        kernel(((self.size + 255) // 256, len(self.parameters)), (256,),
               (device_pointers, np.int32(self.size), np.float32(1 - 0.9 ** self.step),
                np.float32(1 - 0.999 ** self.step), np.float32(0.003), np.float32(1e-8)))
