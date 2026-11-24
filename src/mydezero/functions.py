"""Differentiable NumPy operations and optimization benchmark functions."""

import numpy as np
from mydezero.core import Function, as_variable, as_array, Variable
from mydezero import utils, cuda
import mydezero


# =============================================================================
# Basic Two Functions: square, exp
# =============================================================================

class Square(Function):
    """Compute the elementwise square with a differentiable backward pass."""

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * gy * x
        return gx


def square(x):
    """Compute the elementwise square.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Square()(x)


class Exp(Function):
    """Compute the elementwise exponential with a differentiable backward pass.

    Uses differentiable operations in backward to preserve higher-order gradients.
    """

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = exp(x) * gy
        return gx


def exp(x):
    """Compute the elementwise exponential.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Exp()(x)


# =============================================================================
# Test functions for optimization
# =============================================================================

def sphere(x, y):
    """Evaluate the sphere benchmark for optimization examples.

    Args:
        x (Variable or np.ndarray): First coordinate.
        y (Variable or np.ndarray): Second coordinate.

    Returns:
        Variable or np.ndarray: Benchmark value, with the input wrapper type
            preserved and elementwise evaluation supported.
    """
    return x ** 2 + y ** 2


def matyas(x, y):
    """Evaluate the Matyas benchmark for optimization examples.

    Args:
        x (Variable or np.ndarray): First coordinate.
        y (Variable or np.ndarray): Second coordinate.

    Returns:
        Variable or np.ndarray: Benchmark value, with the input wrapper type
            preserved and elementwise evaluation supported.
    """
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def goldstein(x, y):
    """Evaluate the Goldstein-Price benchmark for optimization examples.

    Args:
        x (Variable or np.ndarray): First coordinate.
        y (Variable or np.ndarray): Second coordinate.

    Returns:
        Variable or np.ndarray: Benchmark value, with the input wrapper type
            preserved and elementwise evaluation supported.
    """
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


# =============================================================================
# Basic functions: sin, cos, tanh
# =============================================================================

class Sin(Function):
    """Compute the elementwise sine with a differentiable backward pass."""

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    """Compute the elementwise sine.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Sin()(x)


class Cos(Function):
    """Compute the elementwise cosine with a differentiable backward pass."""

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    """Compute the elementwise cosine.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Cos()(x)


class Tanh(Function):
    """Compute the elementwise hyperbolic tangent with a differentiable backward pass."""

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    """Compute the elementwise hyperbolic tangent.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Tanh()(x)


# =============================================================================
# Tensor Operations: reshape, transpose, get_item
# =============================================================================
class Reshape(Function):
    """Change an input shape without changing its elements.

    Args:
        shape (tuple[int, ...]): Target shape.

    Attributes:
        shape (tuple[int, ...]): Requested output shape.
        x_shape (tuple[int, ...]): Input shape saved during forward for backward.
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    """Reshape an input while retaining its gradient connection.

    Args:
        x (Variable or np.ndarray): Input values.
        shape (tuple[int, ...]): Target shape with at most one inferred dimension (-1).

    Returns:
        Variable: Reshaped input, or the original variable if its shape matches.
    """
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    """Permute axes and apply the inverse permutation during backward.

    Args:
        axes (tuple[int, ...] or list[int] or None): Axis permutation. None reverses
            all axes. Negative axes are supported.

    Attributes:
        axes (tuple[int, ...] or list[int] or None): Requested axis order.
    """

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axs = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axs)


def transpose(x, axes=None):
    """Permute input axes.

    Args:
        x (Variable or np.ndarray): Input values.
        axes (tuple[int, ...] or list[int] or None): Axis permutation; None reverses
            all axes.

    Returns:
        Variable: Input with its axes reordered.
    """
    return Transpose(axes)(x)


class GetItem(Function):
    """Select elements and scatter their gradients back to the input.

    Args:
        slices (object): A NumPy index, including slices, integers, index arrays,
            boolean masks, or tuples combining indices.
    """

    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    """Accumulate indexing gradients, including repeated indices.

    Uses np.add.at on CPU or scatter_add on GPU so repeated selections
    contribute once per occurrence. Its backward pass gathers elements to
    support higher-order derivatives.

    Args:
        slices (object): Index used in the corresponding GetItem operation.
        in_shape (tuple[int, ...]): Shape of the original input.
    """

    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = mydezero.cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    """Select elements using NumPy indexing while preserving the gradient graph.

    Args:
        x (Variable or np.ndarray): Input values.
        slices (object): NumPy-compatible index. Repeated indices accumulate
            gradients during backward propagation.

    Returns:
        Variable: Selected elements with the shape produced by NumPy indexing.
    """
    f = GetItem(slices)
    return f(x)


# =============================================================================
# sum, sum_to, broadcast_to, matmul, linear
# =============================================================================

class Sum(Function):
    """Reduce an input and broadcast its gradient back to the input shape.

    Args:
        axis (int or tuple[int, ...] or None): Axes to reduce; None reduces all.
        keepdims (bool): Retain reduced axes as dimensions of size one.

    Attributes:
        axis (int or tuple[int, ...] or None): Reduction axes.
        keepdims (bool): Whether reduced dimensions are retained.
        x_shape (tuple[int, ...]): Input shape saved during forward.
    """

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    """Sum elements along the requested axes.

    Args:
        x (Variable or np.ndarray): Input values.
        axis (int or tuple[int, ...] or None): Axes to reduce; None reduces all.
        keepdims (bool): Retain reduced axes as dimensions of size one.

    Returns:
        Variable: Reduced sum.
    """
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    """Reduce broadcast dimensions to a target shape.

    Args:
        shape (tuple[int, ...]): Target shape.

    Attributes:
        shape (tuple[int, ...]): Requested output shape.
        x_shape (tuple[int, ...]): Input shape saved during forward for backward.
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    """Sum elements over broadcast dimensions to obtain a target shape.

    Args:
        x (Variable or np.ndarray): Input values.
        shape (tuple[int, ...]): Target shape.

    Returns:
        Variable: Reduced input, or the original variable if its shape matches.
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    """Broadcast an input to a target shape.

    Args:
        shape (tuple[int, ...]): Target shape.

    Attributes:
        shape (tuple[int, ...]): Requested output shape.
        x_shape (tuple[int, ...]): Input shape saved during forward for backward.
    """

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = mydezero.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    """Broadcast an input to a compatible target shape.

    Args:
        x (Variable or np.ndarray): Input values.
        shape (tuple[int, ...]): Target shape.

    Returns:
        Variable: Broadcast input, or the original variable if its shape matches.
    """
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    """Multiply two matrices and propagate gradients to both operands."""

    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    """Multiply two two-dimensional matrices.

    Args:
        x (Variable or np.ndarray): Left matrix of shape (N, K).
        W (Variable or np.ndarray): Right matrix of shape (K, M).

    Returns:
        Variable: Matrix product of shape (N, M).
    """
    return MatMul()(x, W)


class Linear(Function):
    """Compute an affine transform and gradients for the input, weights, and bias.

    The bias is optional. Its backward gradient is reduced over the batch dimension.
    This Function performs the computation; layers.Linear owns trainable parameters.
    """

    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    """Apply an affine transform to a batch of inputs.

    Args:
        x (Variable or np.ndarray): Input matrix of shape (N, in_size).
        W (Variable or np.ndarray): Weight matrix of shape (in_size, out_size).
        b (Variable or np.ndarray or None): Optional bias of shape (out_size,).

    Returns:
        Variable: Transformed batch of shape (N, out_size).
    """
    return Linear()(x, W, b)


# =============================================================================
# loss_functions: mean_squared_error, softmax_cross_entropy
# =============================================================================

class MeanSquaredError(Function):
    """Sum squared errors and normalize by the batch size, following DeZero.

    Inputs should have the same shape and at least one dimension. For multicolumn
    outputs, the loss sums over output features rather than averaging over them.
    """

    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    """Compute squared error summed over elements and divided by batch size.

    Args:
        x0 (Variable or np.ndarray): Predictions of shape (N, ...) with N > 0.
        x1 (Variable or np.ndarray): Targets with the same shape as x0.

    Returns:
        Variable: Scalar loss normalized by N, not by the total element count.
    """
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    """Compute the mean cross entropy from logits and integer class labels.

    Center logits before logsumexp to avoid overflow and loss of precision from
    large common offsets. Backward differentiates logits through softmax,
    preserving higher-order gradients; labels receive no gradient.
    """

    def forward(self, x, t):
        N = x.shape[0]
        self.gpu_probabilities = None
        if (cuda.gpu_enable and isinstance(x, cuda.cupy.ndarray)
                and isinstance(t, cuda.cupy.ndarray) and x.ndim == 2
                and x.dtype == np.float32 and x.shape[0] > 0 and x.shape[1] > 0
                and t.size == N and t.dtype.kind in 'iu'):
            xp = cuda.cupy
            values = xp.ascontiguousarray(x)
            labels = xp.ascontiguousarray(t, dtype=xp.int64)
            losses = xp.empty(N, dtype=xp.float32)
            self.gpu_probabilities = xp.empty_like(values)
            _cross_entropy_forward((N,), (128,),
                                   (values, labels, np.int32(x.shape[1]),
                                    losses, self.gpu_probabilities))
            return losses.mean()
        shifted = x - x.max(axis=1, keepdims=True)
        log_p = shifted - utils.logsumexp(shifted, axis=1)
        return -log_p[np.arange(N), t.ravel()].mean()

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        if self.gpu_probabilities is not None and not mydezero.Config.enable_backprop:
            xp = cuda.cupy
            labels = xp.ascontiguousarray(t.data, dtype=xp.int64)
            return Variable(_cross_entropy_backward(
                self.gpu_probabilities, labels, gy.data, CLS_NUM, N))

        xp = cuda.get_array_module(t.data)
        gy = gy / xp.array(N, dtype=x.dtype)
        y = softmax(x)

        t_onehot = xp.eye(CLS_NUM, dtype=x.dtype)[t.data.ravel()]
        y = (y - t_onehot) * gy
        return y


if cuda.gpu_enable:
    _cross_entropy_forward = cuda.cupy.RawKernel(r'''
    extern "C" __global__ void mydezero_cross_entropy(
        const float* x, const long long* labels, int classes, float* loss, float* probabilities) {
        __shared__ float scratch[128];
        int row = blockIdx.x, tid = threadIdx.x;
        long long offset = (long long)row * classes;
        float maximum = -__int_as_float(0x7f800000);
        for (int j = tid; j < classes; j += 128) maximum = fmaxf(maximum, x[offset + j]);
        scratch[tid] = maximum;
        __syncthreads();
        for (int step = 64; step; step /= 2) {
            if (tid < step) scratch[tid] = fmaxf(scratch[tid], scratch[tid + step]);
            __syncthreads();
        }
        maximum = scratch[0];
        float sum = 0;
        for (int j = tid; j < classes; j += 128) sum += expf(x[offset + j] - maximum);
        scratch[tid] = sum;
        __syncthreads();
        for (int step = 64; step; step /= 2) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            __syncthreads();
        }
        sum = scratch[0];
        for (int j = tid; j < classes; j += 128)
            probabilities[offset + j] = expf(x[offset + j] - maximum) / sum;
        if (tid == 0) {
            long long label = labels[row];
            if (label < 0) label += classes;
            loss[row] = label >= 0 && label < classes
                ? logf(sum) - (x[offset + label] - maximum) : __int_as_float(0x7fc00000);
        }
    }
    ''', 'mydezero_cross_entropy')
    _cross_entropy_backward = cuda.cupy.ElementwiseKernel(
        'float32 probability, raw int64 label, S scale, int32 classes, int32 batch',
        'S gx',
        '''
        long long target = label[i / classes];
        if (target < 0) target += classes;
        gx = (probability - (i % classes == target)) * (scale / batch);
        ''', 'mydezero_cross_entropy_backward')


def softmax_cross_entropy(x, t):
    """Compute batch-averaged softmax cross entropy from unnormalized logits.

    Args:
        x (Variable or np.ndarray): Finite floating-point logits of nonempty
            shape (N, C). Do not apply softmax before passing x.
        t (Variable or np.ndarray): Integer class indices of shape (N,) or
            (N, 1), each in the range [0, C). One-hot labels are not supported.

    Returns:
        Variable: Scalar mean negative log probability of the target classes.
    """
    return SoftmaxCrossEntropy()(x, t)


# =============================================================================
# accuracy
# =============================================================================
def accuracy(y, t):
    """Return mean classification accuracy without a differentiation graph.

    Args:
        y (Variable or np.ndarray): Nonempty scores of shape (N, C). Ties select
            the first maximum, following NumPy argmax.
        t (Variable or np.ndarray): Class indices of shape (N,) or (N, 1).

    Returns:
        Variable: Scalar fraction of correct predictions; not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

def dropout(x, dropout_ratio=0.5):
    """Randomly disable elements to prevent overfitting.

    In training mode, each element is dropped with probability
    ``dropout_ratio`` and the remaining values are scaled by
    ``1 / (1 - dropout_ratio)`` so the expected sum stays unchanged. In
    `test_mode`, the input is returned unchanged.

    Args:
        x (Variable or np.ndarray): Input values of any shape.
        dropout_ratio (float): Drop probability in [0, 1) per element.

    Returns:
        Variable: Masked and scaled input, or the unchanged input in
            `test_mode`.
    """
    x = as_variable(x)

    if mydezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x

# =============================================================================
# batch normalization
# =============================================================================
class BatchNorm(Function):
    """Normalize channel values using batch or running statistics."""

    def __init__(self, avg_mean, avg_var, decay, eps):
        self.avg_mean = avg_mean
        self.avg_var = avg_var
        self.decay = decay
        self.eps = eps

    def forward(self, x, gamma, beta):
        if x.ndim not in (2, 4) or x.size == 0:
            raise ValueError("batch_norm expects a nonempty (N, C) or (N, C, H, W) input")
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError("batch_norm expects floating-point input")

        channels = x.shape[1]
        if any(p.shape != (channels,) for p in (gamma, beta, self.avg_mean, self.avg_var)):
            raise ValueError("batch_norm expects one scale, bias, mean, and variance per channel")

        self.axes = (0,) if x.ndim == 2 else (0, 2, 3)
        self.channel_shape = (1, channels) + (1,) * (x.ndim - 2)
        self.count = x.size // channels
        self.train = mydezero.Config.train
        xp = cuda.get_array_module(x)

        if self.train:
            mean = x.mean(axis=self.axes)
            var = x.var(axis=self.axes)
            # Keep the running variance convention used by DeZero.
            correction = self.count / (self.count - 1) if self.count > 1 else 1
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += ((1 - self.decay) * correction) * var
        else:
            mean, var = self.avg_mean, self.avg_var
            self.inference_mean = mean.copy().reshape(self.channel_shape)

        self.inv_std = (1 / xp.sqrt(var + self.eps)).reshape(self.channel_shape)
        self.normalized = (x - mean.reshape(self.channel_shape)) * self.inv_std
        return self.normalized * gamma.reshape(self.channel_shape) + beta.reshape(self.channel_shape)

    def backward(self, gy):
        x, gamma, _ = self.inputs
        normalized, inv_std = self.normalized, self.inv_std
        if mydezero.Config.enable_backprop:
            # Rebuild the dependency on x when recording a gradient graph.
            if self.train:
                centered = x - x.sum(axis=self.axes, keepdims=True) / self.count
                var = (centered ** 2).sum(axis=self.axes, keepdims=True) / self.count
                inv_std = (var + self.eps) ** -0.5
                normalized = centered * inv_std
            else:
                normalized = (x - self.inference_mean) * inv_std

        gbeta = gy.sum(axis=self.axes)
        ggamma = (gy * normalized).sum(axis=self.axes)
        if self.train:
            gy = gy - (gbeta.reshape(self.channel_shape)
                       + normalized * ggamma.reshape(self.channel_shape)) / self.count
        gx = gy * gamma.reshape(self.channel_shape) * inv_std
        return gx, ggamma, gbeta


def batch_norm(x, gamma, beta, avg_mean, avg_var, decay=0.9, eps=2e-5):
    """Normalize a floating-point (N, C) or (N, C, H, W) batch by channel.

    The running mean and variance arrays are updated in place in training mode
    and used as fixed statistics in test mode. All arrays share a device.
    Gradients are computed for x, gamma, and beta.
    """
    return BatchNorm(avg_mean, avg_var, decay, eps)(x, gamma, beta)


# =============================================================================
# activation function: sigmoid, softmax, ReLU
# =============================================================================

def sigmoid_simple(x):
    """Compute the logistic sigmoid by composing differentiable operations.

    This teaching version constructs the graph for 1 / (1 + exp(-x)); use sigmoid
    for the formulation that avoids exponential overflow on large negative inputs.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise sigmoid with the same shape as x.
    """
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class Sigmoid(Function):
    """Compute the elementwise logistic sigmoid with a differentiable backward pass.

    Uses the tanh formulation from DeZero to avoid exponential overflow.
    """

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    """Compute the elementwise logistic sigmoid.

    Args:
        x (Variable or np.ndarray): Input values.

    Returns:
        Variable: Elementwise result with the same shape as x.
    """
    return Sigmoid()(x)


class Softmax(Function):
    """Normalize exponentials after subtracting the maximum along selected axes.

    Args:
        axis (int, tuple[int, ...], or None): Normalization axes. Defaults to 1.
            Negative axes follow NumPy conventions; None uses all dimensions.
    """

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    """Convert logits to probabilities while preserving the gradient graph.

    Args:
        x (Variable or np.ndarray): Finite floating-point input values.
        axis (int, tuple[int, ...], or None): Normalization axes. Defaults to 1,
            the class axis for an (N, C) batch. Use axis=0 for a 1D input.

    Returns:
        Variable: Probabilities with the same shape as x, summing to one along
            the selected axes.
    """
    return Softmax(axis)(x)

class ReLU(Function):
    """Clamp negative values to zero; use zero derivative at the origin."""

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    """Apply max(x, 0) elementwise, preserving shape and the gradient graph."""
    return ReLU()(x)


# =============================================================================
# max / min / clip
# =============================================================================
class Max(Function):
    """Reduce maxima and send the full gradient to every tied maximum.

    Equality masks define the backward convention at ties; the incoming
    gradient is not divided among equal extrema.

    Args:
        axis (int, tuple[int, ...], or None): Reduction axes. Defaults to None.
        keepdims (bool): Keep reduced dimensions with size one when True.
    """

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    """Reduce minima with the same axis and tied-gradient conventions as Max."""

    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    """Reduce input maxima over the selected axes.

    Args:
        x (Variable or np.ndarray): Input values.
        axis (int, tuple[int, ...], or None): Reduction axes, including negative
            axes. None reduces all dimensions; an empty tuple reduces none.
        keepdims (bool): Keep reduced dimensions with size one when True.

    Returns:
        Variable: Maximum values. Backward sends the full incoming gradient to
            each tied maximum.
    """
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    """Reduce input minima over the selected axes.

    Args:
        x (Variable or np.ndarray): Input values.
        axis (int, tuple[int, ...], or None): Reduction axes, including negative
            axes. None reduces all dimensions; an empty tuple reduces none.
        keepdims (bool): Keep reduced dimensions with size one when True.

    Returns:
        Variable: Minimum values. Backward sends the full incoming gradient to
            each tied minimum.
    """
    return Min(axis, keepdims)(x)


class Clip(Function):
    """Clamp values with gradient one inside and on the interval boundaries.

    Args:
        x_min (float): Lower bound of the clipping interval.
        x_max (float): Upper bound, at least x_min.
    """

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    """Clamp input values to a closed interval without modifying the input.

    Args:
        x (Variable or np.ndarray): Input values.
        x_min (float): Lower bound of the clipping interval.
        x_max (float): Upper bound, at least x_min.

    Returns:
        Variable: Clipped values with the same shape as x. Backward passes the
            gradient through values in [x_min, x_max], including the boundaries,
            and returns zero for values outside the interval.
    """
    return Clip(x_min, x_max)(x)

from mydezero.functions_conv import conv2d
from mydezero.functions_conv import deconv2d
from mydezero.functions_conv import im2col
from mydezero.functions_conv import col2im
from mydezero.functions_conv import pooling
from mydezero.functions_conv import average_pooling
