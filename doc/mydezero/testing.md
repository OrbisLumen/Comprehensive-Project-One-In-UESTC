# Tests and Coverage

Run from the project directory:

```bash
python -m pytest tests/mydezero -q
```

Use the notebook's Python environment. Tests require NumPy, Matplotlib, and pytest.
The rendering test requires system Graphviz and skips without `dot`.
IPython supports notebook image display. Run `python -m pytest tests/mydezero --collect-only -q` to list all cases, including optional GPU tests.

## Feature Coverage

| Feature | Test files | Checks |
| --- | --- | --- |
| Variable, Parameter, core wrappers | `core/test_variable.py`, `nn/test_layers.py`, `nn/test_models.py` | Properties, types, wrapper identity, parameter registration, gradient clearing |
| Operators and broadcasting | `core/test_arithmetic.py` | Forward/backward operations, scalars, arrays, broadcast gradient reduction |
| Automatic differentiation | `core/test_backward.py` | Branches, variable reuse, gradient accumulation, intermediate gradients, custom multi-output functions, finite differences |
| Configuration contexts | `core/test_config.py` | Graph disabling, nesting, restoration after exceptions |
| Elementary functions and activations | `functions/test_elementary.py` | square/exp/sin/cos/tanh, both sigmoids, ReLU at zero, higher-order derivatives, stability |
| Dropout | `core/test_config.py`, `functions/test_elementary.py` | test_mode identity and flag restoration, mask ratio, rescaling, zero ratio, shape and dtype preservation |
| Convolution and pooling | `functions/test_conv.py`, `nn/test_layers.py` | conv2d/deconv2d input/weight/bias numerical gradients, convolution weight second derivatives, max-pooling argmax routing, rectangular average pooling, Conv2d/Deconv2d lazy init and shapes |
| CPU convolution and pooling | `functions/test_cpu_convolution.py`, `functions/test_pooling_inference.py` | Forward and gradient parity with the original NumPy paths, dtype and layout coverage, and pooling edge cases and dispatch |
| Weight initialization dtype | `nn/test_layers.py`, `integration/test_gpu.py` | Declared dtype is preserved in parameters, outputs, and gradients during eager and lazy CPU initialization and lazy CUDA initialization |
| Patch extraction and accumulation | `functions/test_im2col.py` | im2col/col2im values and layout, matrix and 6D output, batches/channels, rectangular kernels, stride/padding, uncovered pixels, noncontiguous input, dtype and input preservation, overlap sums, numerical gradients and second derivatives |
| Optimization examples | `functions/test_optimization.py` | sphere/matyas/goldstein gradients |
| Tensor operations | `functions/test_tensor.py` | reshape, explicit transpose, sum, sum_to, broadcast_to, indexing, repeated indices, higher-order indexing gradients |
| Matrix operations and regression loss | `functions/test_tensor.py`, `nn/test_layers.py` | matmul, linear with/without bias, input/weight/bias gradients, MSE batch normalization |
| Batched matmul and self-attention | `functions/test_batched_matmul.py`, `nn/test_layers.py` | Batched forward parity with NumPy, numerical gradients for both operands, rank and batch validation, known-value attention, single-token and noncontiguous inputs, parameter registration, gradient checks, and weight roundtrip |
| Classification | `functions/test_classification.py` | Softmax axes, cross entropy numerical gradients, second derivatives, large logits, underflow, single class, accuracy, column labels, no-gradient behavior |
| Extrema and clipping | `functions/test_extrema.py` | Positive/negative axes, retained dimensions, ties, boundaries, higher-order gradients |
| Layers and models | `nn/test_layers.py`, `nn/test_models.py` | Deferred initialization, optional bias, recursive parameters, Sequential, MLP, hidden activations, plot calls, weight save/load roundtrip |
| VGG16 preprocessing | `nn/test_models.py` | BGR ordering, mean subtraction, CHW layout, floating-point dtypes, and integer dtype rejection. Requires Pillow |
| Batch normalization | `functions/test_batch_norm.py`, `nn/test_layers.py` | NC and NCHW outputs, DeZero's running-variance correction, updates under training with `no_grad()`, numerical gradients in both modes, higher-order gradients, and weight roundtrip |
| ResNet | `nn/test_models.py` | Supported stage depths and parameter keys, residual downsampling and gradients, global average pooling equivalence, and ResNet50 inference logits |
| Optimizers | `nn/test_optimizers.py` | SGD, MomentumSGD, hooks, independent velocities, missing gradients, zero momentum |
| Dataset and MNIST | `datasets/test_datasets.py` | prepare, unlabeled data, transforms, train/test files, IDX headers, shapes/dtypes, raw storage, sample preview |
| DataLoader | `datasets/test_dataloaders.py` | Order, shuffle, partial batches, repeated epochs, reset, next, empty/unlabeled data |
| Transforms | `datasets/test_transforms.py` | Compose order and empty lists, flattening, casting, normalization, channel broadcasting, input preservation |
| Downloads and progress | `test_downloads.py` | File downloads, nested/default/custom caches, names, URL queries, cache reuse, failure/interruption cleanup, unknown totals, progress limits |
| Numerical and graph utilities | `test_utils.py`, `integration/test_graph_rendering.py` | Stable logsumexp, reduction shapes, DOT generation, actual PNG rendering |
| Training pipeline | `integration/test_training.py` | Model/optimizer combinations, decreasing loss, Dataset → DataLoader → classification training → weighted evaluation |

Paths are relative to `tests/mydezero/`. Tests use known results, analytic gradients, finite
differences, and small training tasks. They check correctness, not just line coverage,
and do not cover every input combination.

## Reproducibility and External Dependencies

MNIST tests generate two-image IDX/GZIP fixtures with real headers and pixels,
then replace the download entry point to test parsing without network access or
user-cache changes. Download tests use temporary `file://` sources and simulate
network failures and Ctrl+C. Shuffle tests inject fixed index orders to check
sample coverage and label alignment.

The full MNIST notebook needs network access or cached data. Its five-epoch training
run is outside the default suite. Automated training tests use small deterministic
data to check parameter updates and accuracy. Preview tests check image pixels and
shape while replacing Matplotlib display calls. Graph rendering separately verifies
a real PNG file. Optional GPU tests require CuPy and a CUDA device; see
[GPU setup](../cupy-setup.md) for installation and execution.
`integration/test_gpu.py` checks CPU/GPU transfers, broadcast gradients,
convolution and transposed-convolution outputs and input/weight/bias gradients,
max/average pooling gradients, BatchNorm gradients and running statistics,
and classifier training with MomentumSGD.
Convolution checks include asymmetric stride and padding and exercise the
GPU im2col/col2im kernels. Missing CuPy/devices skip these tests; kernel or
library failures on an available GPU fail. This coverage does not certify
every GPU operation, dtype, or hardware configuration.

The [GPU optimization notebook](../../examples/neural_network/gpu_optimization.ipynb)
checks fused CUDA Adam against the standard update over successive steps and
records synchronized P50 and P95 for the update and complete training steps.
It also covers pointwise convolution, direct pooling, fused cross entropy,
paired CNN training and cumulative ablation. The standalone cuDNN probe and
equal-size multi-tensor Adam helper remain example feasibility experiments.
`integration/test_gpu_optimized.py` checks pointwise gradients, pooling edge
semantics, loss stability, nondefault-stream execution and higher derivatives.
It also verifies FP32 gradients through BatchNorm and global average pooling.
CPU tests cover counts above the FP16 finite range for both operations.
`integration/test_cudnn.py` checks optional cuDNN forward and gradients,
higher derivatives, weight updates, noncontiguous input and nondefault streams.
Follow the [optional cuDNN setup](../cupy-setup.md#optional-cudnn-9) to run these
tests.
The large-model notebook follow-up reports official-weight inference, short
synthetic training steps, dtype ablation and independent process repetitions.
These throughput tests establish no dataset-level training quality claim.

ResNet tests use random initialization and require no downloads. The default
suite does not load official pretrained archives. The
[pretrained CNN validation notebook](../../examples/neural_network/pretrained_cnn_validation.ipynb)
provides an opt-in comparison of VGG16 and ResNet50 logits with local DeZero
using the official archives. Its offline pooling benchmark checks forward and
gradient equivalence across eight tensor shapes and reports CPU latency and
synchronized CUDA latency when CuPy and a CUDA device are available.
The separate ResNet50 benchmark uses the same official weights and FP32
`(1, 3, 224, 224)` input for both pooling paths. It checks logits and reports
the median and range of 11 warmed-up forward passes per path, alternating their
order. Timed runs exclude downloading, weight loading, image preprocessing,
and device transfers. CUDA timings include a stream synchronization per
forward pass. The full-model benchmark runs with MyDeZero alone.

The [CPU optimization notebook](../../examples/neural_network/cpu_optimization.ipynb)
checks weight initialization dtype and measures convolution and inference max
pooling. It compares the old initializer's dtype with FP32, then isolates
operator contributions with eight FP32 combinations on Toy, Iris, and MNIST.
The notebook records P50, P95, throughput, tracked operator allocations, and
held-out classification metrics. It also reports PyTorch CPU timings.

Run Python snippets in each document from top to bottom. MNIST examples in
`data.md` download and load real data; `show()` displays images. When adding APIs,
update [api.md](api.md), relevant usage notes, and this table, and add behavior tests.
