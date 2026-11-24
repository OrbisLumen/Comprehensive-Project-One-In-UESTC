# Usage and Implementation Notes

See the [complete API index](api.md), [datasets and mini-batches](data.md),
and [test coverage guide](testing.md) for the rest of the current framework.

## Indexing, Composed Models, and Momentum

Use `get_item` for differentiable NumPy indexing:

```python
import numpy as np
from mydezero import Variable
import mydezero.functions as F

x = Variable(np.array([2.0, 3.0, 5.0]))
y = F.get_item(x, [0, 0, 2])
y.sum().backward()
np.testing.assert_array_equal(x.grad.data, [2.0, 0.0, 1.0])
```

Repeated indices accumulate gradients at the selected positions; unselected
positions receive zero. Use `backward(create_graph=True)` when computing
higher-order derivatives. Call `F.get_item(x, indices)` explicitly; `Variable`
currently has no `x[indices]` operator.

`Sequential` applies its layers in order and registers them for parameter
collection and gradient clearing. An empty Sequential returns its input.
`MLP` accepts a nonempty sequence of output widths, infers input widths on its
first call, and applies sigmoid between hidden layers by default. Its last layer
has no activation. Pass `activation=F.tanh` to select another differentiable
hidden-layer activation.

```python
import numpy as np
import mydezero.functions as F
from mydezero.layers import Linear
from mydezero.models import MLP, Sequential
from mydezero.optimizers import MomentumSGD

# Two affine layers; Sequential does not insert activations.
sequence = Sequential(Linear(4), Linear(1))
assert sequence(np.zeros((2, 3), dtype=np.float32)).shape == (2, 1)

# One hidden layer with sigmoid and one affine output layer.
model = MLP((4, 1))
optimizer = MomentumSGD(lr=0.01, momentum=0.9).setup(model)
inputs = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
targets = np.array([[1.0], [3.0], [5.0]], dtype=np.float32)

for _ in range(100):
    model.cleargrads()
    loss = F.mean_squared_error(model(inputs), targets)
    loss.backward()
    optimizer.update()
```

MomentumSGD stores one velocity per parameter, initialized with zeros on its
first update. It computes `v = momentum * v - lr * grad`, then `param += v`.
Reuse the optimizer across training steps to retain momentum; creating a new
optimizer resets its state. Parameters without gradients are skipped, retaining
their velocity. Hooks run before updates, as with SGD. A momentum of zero produces
the same updates as SGD. Trainable parameters should use floating-point NumPy
arrays.

## Softmax and Classification Loss

`F.softmax(x, axis=1)` converts an `(N, C)` batch of logits into class
probabilities. It subtracts the maximum before exponentiation to avoid overflow.
For a 1D input, specify `axis=0`. Other NumPy axes, including negative axes,
tuples, and `None`, are also supported.

Pass unnormalized logits directly to `F.softmax_cross_entropy(x, t)`.
The loss centers logits before computing log probabilities and returns the mean
negative log probability over the batch. Logits must have nonempty shape `(N, C)`;
labels must be integer class indices in `[0, C)` with shape `(N,)` or `(N, 1)`.
One-hot labels are not accepted. Backward computes `(softmax(x) - one_hot(t)) / N`
for logits; labels receive no gradient. Floating-point logits should be finite.

```python
import numpy as np
from mydezero import Variable
import mydezero.functions as F

logits = Variable(np.zeros((2, 3), dtype=np.float32))
labels = np.array([[0], [2]], dtype=np.int64)
loss = F.softmax_cross_entropy(logits, labels)
loss.backward()

np.testing.assert_allclose(loss.data, np.log(3), rtol=1e-6)
assert logits.grad.shape == (2, 3)
assert logits.grad.dtype == np.float32
probabilities = F.softmax(logits)
np.testing.assert_allclose(probabilities.data.sum(axis=1), 1)
```

Use the same `model.cleargrads()`, `loss.backward()`, and `optimizer.update()`
training steps as for squared error, replacing the loss with cross entropy and
setting the final model width to the number of classes. Call
`backward(create_graph=True)` to retain a differentiable gradient graph for
second-order derivatives.

## Extrema, Clipping, and Numerical Helpers

`F.max` and `F.min` accept `axis` and `keepdims` with NumPy semantics. Negative
axes and tuples are supported. Their backward convention sends the full incoming
gradient to every tied extremum rather than dividing it among ties.

`F.clip(x, x_min, x_max)` clamps values using NumPy without modifying its input.
The gradient passes through values inside the interval and exactly on either
boundary, and is zero outside it.

```python
import numpy as np
from mydezero import Variable
import mydezero.functions as F

x = Variable(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
np.testing.assert_array_equal(F.max(x, axis=-1).data, [2.0])
np.testing.assert_array_equal(F.min(x, axis=-1).data, [-2.0])
clipped = F.clip(x, -1.0, 1.0)
clipped.sum().backward()
np.testing.assert_array_equal(clipped.data, [[-1, -1, 0, 1, 1]])
np.testing.assert_array_equal(x.grad.data, [[0, 1, 1, 1, 0]])
```

`utils.logsumexp(x, axis=1)` computes log-sum-exp on a NumPy array while retaining
reduced dimensions. It avoids exponential overflow and preserves the input, but
does not record an automatic differentiation graph. `utils.max_backward_shape`
restores reduced axes as size one for broadcasting max/min gradients.

## VGG16 Inference

Use Pillow to open the image. The default preprocessing produces a
`(3, 224, 224)` array in BGR order with the channel means subtracted.
Add a batch dimension before calling the model.

```python
from PIL import Image
from mydezero import no_grad, test_mode
from mydezero.models import VGG16

model = VGG16(pretrained=True)
with Image.open("image.jpg") as image:
    x = VGG16.preprocess(image)[None, ...]
with test_mode(), no_grad():
    logits = model(x)
assert logits.shape == (1, 1000)
```

`pretrained=True` fetches the DeZero weights on first use and reuses the
cached file. For a compatible local archive, use `model = VGG16()` followed
by `model.load_weights("models/vgg16.npz")`. Use the default image size with
the official weights so that the flattened features match the first dense
layer. `test_mode()` disables dropout and `no_grad()` disables graph recording.

## Batch Normalization

`BatchNorm()` normalizes each channel of a floating-point `(N, C)` or
`(N, C, H, W)` batch. The function reduces over `N` for matrices and over
`N`, `H`, and `W` for images. This gives the same per-channel statistics as
DeZero's reshape to `(N * H * W, C)`.

Training normalizes by the batch mean and population variance. Both running
statistics are updated in place as `decay * running + (1 - decay) * batch`.
For the running variance update, the batch variance is first multiplied by
`m / max(m - 1, 1)`, where `m` is the number of values per channel. This
follows DeZero's update, including batches with one or two values per channel.

`test_mode()` uses fixed running statistics and leaves them unchanged.
MyDeZero supports differentiation in both modes, including higher-order
derivatives when using `backward(create_graph=True)`. `no_grad()` controls graph
recording independently. A training forward pass still updates the running
statistics inside `no_grad()`. Use both contexts for inference. Weight archives
include `avg_mean`, `avg_var`, `gamma`, and `beta`.

## ResNet Inference

Use `ResNet50`, `ResNet101`, or `ResNet152` from `mydezero.models`. Provide a
prepared floating-point NCHW batch `x`. For pretrained inference, use the image
preprocessing expected by the weights.

```python
from mydezero import no_grad, test_mode
from mydezero.models import ResNet50

model = ResNet50(pretrained=True)
with test_mode(), no_grad():
    logits = model(x)
assert logits.shape == (x.shape[0], 1000)
```

For a compatible local archive, use `ResNet50()` and call
`model.load_weights("models/resnet50.npz")`. If running on CUDA, transfer the
model and input to the GPU after loading the weights.

VGG16 and ResNet inference run entirely in MyDeZero. The official DeZero release
provides the weight files. `pretrained=True` downloads or reuses a cached
`.npz` archive, and `load_weights(path)` reads a compatible local archive by
parameter name with NumPy. A local archive works offline. The notebook's
optional reference comparison imports DeZero from a local checkout.

## Self-attention

`layers.SelfAttention` applies single-head scaled dot-product self-attention to
an `(N, T, C)` token batch. It owns query, key, and value projections and returns
an `(N, T, out_size)` batch. Masks, dropout, multi-head splits, and returned
attention weights are not implemented. `F.batched_matmul` multiplies equal-shaped
batches over their last two axes and is also usable on its own.

```python
import numpy as np
from mydezero import Variable
from mydezero.layers import SelfAttention

attention = SelfAttention(out_size=8, in_size=4, dtype=np.float64)
x = Variable(np.random.default_rng(0).standard_normal((2, 5, 4)))
y = attention(x)
y.sum().backward()
assert y.shape == (2, 5, 8)
assert attention.q_proj.W.grad is not None
```

## Implementation Compared with DeZero

The reference implementation is `../../reference/deep-learning-from-scratch-3-master-cn/dezero`.
See [the DeZero license](dezero-license.md) for the source attribution.
MyDeZero currently implements a NumPy subset of that framework:

| Module | Current implementation | Scope compared with DeZero |
| --- | --- | --- |
| `core.py` | Variable, Parameter, Function, graph configuration, arithmetic, higher-order differentiation | Uses a generation heap for backward traversal and supports NumPy and optional CuPy arrays |
| `functions.py` | Elementary functions, tensor operations, losses, activations, batch normalization, and batched matrix multiplication | NumPy indexing accumulates gradients for repeated indices. Matrix multiplication and linear transforms support 2D batches, while `batched_matmul` multiplies equal-shaped batches over their last two axes. Self-attention is not part of the local DeZero reference. Classification loss accepts integer labels and averages over the batch. Batch normalization reduces directly over the non-channel axes |
| `layers.py` | Recursive parameter collection, Linear, Conv2d, Deconv2d, BatchNorm, SelfAttention, and weight serialization | Supports lazy initialization and DeZero's hierarchical parameter keys; SelfAttention is a single-head extension without masks, dropout, or multi-head splits |
| `models.py` | Model base class, graph visualization, Sequential, MLP, VGG16, and ResNet50/101/152 | VGG16 and ResNet support DeZero pretrained weights |
| `optimizers.py` | Optimizer base class, hooks, SGD, and MomentumSGD | Other update rules and built-in hooks remain unimplemented |
| `datasets.py` | Dataset base class and cached MNIST loading, raw image preview | Entire split is loaded in memory; transforms run per sample |
| `dataloaders.py` | Shuffled or sequential NumPy mini-batches | Includes partial final batches and resets after exhaustion |
| `transforms.py` | Compose, Flatten, Normalize, AsType/ToFloat, ToInt | NumPy sample transforms; channel-first normalization |
| `utils.py` | DOT graph generation, rendering, reduction gradient helpers, stable logsumexp, downloads and progress | Downloads use a local disk cache; numerical helpers use NumPy; `graph.py` is currently empty |

Library docstrings use Google style (`Args`, `Returns`, `Yields`, `Attributes`, and
`Raises` where relevant). Function subclasses inherit the forward/backward contract
from `Function`; simple overrides document the operation at the class level.

Assign parameters and child layers directly to attributes so `Layer.params()` can
collect them. Gradients are `Variable` objects; call `model.cleargrads()` before each
new backward pass and attach the optimizer with `SGD(...).setup(model)` before
calling `update()`.

Graph rendering requires the Graphviz `dot` executable. Rendered files and shared
temporary DOT source are stored in `draft/`; notebook display also uses IPython.
