"""Trainable layers and recursive parameter collection."""

import math
import os
import weakref

import numpy as np
from mydezero import Parameter, cuda
import mydezero.functions as F
from mydezero.utils import pair

class Layer:
    """Base class for layers that own parameters or other layers.

    Assign Parameter and Layer objects directly to attributes to register them.
    Parameters stored only in ordinary lists or dictionaries are not collected.

    Attributes:
        inputs (list[weakref.ReferenceType]): Weak references to the latest inputs.
        outputs (list[weakref.ReferenceType]): Weak references to the latest outputs.
    """

    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        """Register parameters and child layers when assigning attributes.

        Args:
            name (str): Attribute name.
            value (object): Value to assign.
        """
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        """Run forward and retain weak references to the latest inputs and outputs.

        Args:
            *inputs (Variable or np.ndarray): Inputs accepted by the subclass.

        Returns:
            Variable or list[Variable]: Single output or a list of outputs.
        """
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        """Compute layer outputs; subclasses provide their own input signatures.

        Args:
            inputs (Variable or np.ndarray): Layer input. Subclasses may accept
                multiple positional inputs.

        Returns:
            Variable or tuple[Variable, ...]: One or more output variables.

        Raises:
            NotImplementedError: If a subclass does not implement forward.
        """
        raise NotImplementedError

    def params(self):
        """Iterate through registered parameters, including those in child layers.

        Traversal order is unspecified because attribute names are stored in a set.

        Yields:
            Parameter: A registered parameter from this layer or a child layer.
        """
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        """Reset gradients on all registered parameters recursively."""
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        """Move all parameters to the CPU."""
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        """Move all parameters to the GPU."""
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        """Collect parameters of this layer and its child layers into a dict.

        Nested layers contribute entries under names joined with ``'/'``,
        producing keys such as ``'l0/W'`` that stay unique across the model
        and match the keys written by `save_weights`.

        Args:
            params_dict (dict): Dictionary updated in place with parameter
                name keys and Parameter values.
            parent_key (str): Name prefix for nested layers; empty for the
                top-level layer.
        """
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        """Save all parameters as a compressed .npz archive.

        Parameter names are flattened with ``'/'`` separators (see
        `_flatten_params`) so nested layers are preserved. The layer is
        moved to the CPU before saving. If saving fails, a partially
        written file is removed and the error is re-raised.

        Args:
            path (str): File path to save to. Use the ``.npz`` extension so
                the same path can be passed to `load_weights`.
        """
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        """Load parameters saved by `save_weights` into this layer.

        Each parameter's data is replaced with the value stored under the
        matching flattened name, so the layer structure must match the
        saved archive. Loaded data is on the CPU; call `to_gpu()`
        afterwards when training on the GPU.

        Args:
            path (str): File path to a .npz archive written by
                `save_weights`.
        """
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

class Linear(Layer):
    """Fully connected layer with optional bias and deferred weight initialization.

    Args:
        out_size (int): Number of output features.
        nobias (bool): Omit the bias when True.
        dtype (np.dtype or type): Data type for initialized parameters.
        in_size (int or None): Number of input features. None defers weight
            initialization until the first forward pass.

    Attributes:
        in_size (int or None): Input width, inferred on the first pass if omitted.
        out_size (int): Output width.
        dtype (np.dtype or type): Parameter initialization data type.
        W (Parameter): Weight matrix of shape (in_size, out_size), initially
            holding None when initialization is deferred.
        b (Parameter or None): Bias vector of shape (out_size,), or None.
    """

    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self, xp = np):
        """Initialize weights with standard deviation sqrt(1 / in_size).

        Args:
            xp (module): NumPy or CuPy module used to sample the weights on
                the device of the input.
        """
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O) * np.sqrt(1 / I)
        self.W.data = W_data.astype(self.dtype, copy=False)

    def forward(self, x):
        """Apply the affine transform, initializing weights once if needed.

        Args:
            x (Variable or np.ndarray): Input batch of shape (N, in_size).

        Returns:
            Variable: Output batch of shape (N, out_size).
        """
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        """Initialize filters with standard deviation sqrt(1 / (C * KH * KW)).

        Args:
            xp (module): NumPy or CuPy module used to sample the filters on
                the device of the input.
        """
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW) * scale
        self.W.data = W_data.astype(self.dtype, copy=False)

    def forward(self, x):
        """Apply the convolution, initializing filters once if needed.

        Args:
            x (Variable or np.ndarray): Input batch of shape (N, C, H, W).

        Returns:
            Variable: Output batch of shape (N, out_channels, OH, OW).
        """
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution) layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        """Initialize filters with standard deviation sqrt(1 / (C * KH * KW)).

        Args:
            xp (module): NumPy or CuPy module used to sample the filters on
                the device of the input.
        """
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW) * scale
        self.W.data = W_data.astype(self.dtype, copy=False)

    def forward(self, x):
        """Apply the transposed convolution, initializing filters once if needed.

        Args:
            x (Variable or np.ndarray): Input batch of shape (N, C, H, W).

        Returns:
            Variable: Output batch of shape (N, out_channels, OH, OW).
        """
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y

# =============================================================================
# BatchNorm
# =============================================================================


class BatchNorm(Layer):
    """Normalize channels of a floating-point (N, C) or (N, C, H, W) batch."""

    def __init__(self):
        super().__init__()
        # Register running statistics for weight serialization and device transfers.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        if x.ndim not in (2, 4) or x.size == 0:
            raise ValueError("BatchNorm expects a nonempty (N, C) or (N, C, H, W) input")
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError("BatchNorm expects floating-point input")

        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def forward(self, x):
        """Apply normalization and lazily initialize state on the input device."""
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_norm(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)


# =============================================================================
# SelfAttention
# =============================================================================

class SelfAttention(Layer):
    """Single-head scaled dot-product self-attention over a token sequence batch.

    Each token is projected to queries, keys, and values by separate Linear
    layers. Attention weights are softmax(Q K^T / sqrt(d_k)) along the key axis.
    Masks, dropout, multi-head splits, and returned attention weights are not
    implemented.

    Args:
        out_size (int): Width of the query, key, value, and output projections.
        in_size (int or None): Width of each input token. None defers parameter
            initialization until the first forward pass.
        dtype (np.dtype or type): Data type for initialized parameters.

    Attributes:
        out_size (int): Projection width, used as d_k in the scale.
        in_size (int or None): Input token width, inferred on the first pass.
        dtype (np.dtype or type): Parameter initialization data type.
        q_proj (Layer): Query projection.
        k_proj (Layer): Key projection.
        v_proj (Layer): Value projection.

    Raises:
        ValueError: If the input is not a three-dimensional (N, T, C) batch
            or the token axis is empty.
    """

    def __init__(self, out_size, in_size=None, dtype=np.float32):
        super().__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.q_proj = Linear(out_size, dtype=dtype, in_size=in_size)
        self.k_proj = Linear(out_size, dtype=dtype, in_size=in_size)
        self.v_proj = Linear(out_size, dtype=dtype, in_size=in_size)

    def forward(self, x):
        """Apply self-attention to a nonempty batch of nonempty token sequences.

        Args:
            x (Variable or np.ndarray): Input batch of shape (N, T, in_size), with
                N > 0 and T > 0.

        Returns:
            Variable: Attended batch of shape (N, T, out_size).

        Raises:
            ValueError: If the input does not have exactly three dimensions
                or the token axis is empty.
        """
        if x.ndim != 3:
            raise ValueError('SelfAttention expects an (N, T, C) input')
        batch, length = x.shape[0], x.shape[1]
        if length == 0:
            raise ValueError('SelfAttention expects at least one token per sequence')
        if self.in_size is None:
            self.in_size = x.shape[2]
        flat = x.reshape(batch * length, self.in_size)
        q = self.q_proj(flat).reshape(batch, length, self.out_size)
        k = self.k_proj(flat).reshape(batch, length, self.out_size)
        v = self.v_proj(flat).reshape(batch, length, self.out_size)
        scores = F.batched_matmul(q, F.transpose(k, (0, 2, 1)))
        scores = scores * (1.0 / math.sqrt(self.out_size))
        weights = F.softmax(scores, axis=-1)
        return F.batched_matmul(weights, v)
