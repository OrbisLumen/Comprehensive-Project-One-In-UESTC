# Jupyter Examples

From the repository root, install the library in the notebook kernel's Python environment:

```bash
python -m pip install -e .
```

Examples use NumPy. GPU examples additionally require CuPy and a CUDA device.
Plots require Matplotlib; computation graphs require IPython and system Graphviz
(`dot`). Select the matching Python 3 kernel and run cells in order.

Suggested reading order:

| Category | Notebook | Topics |
| --- | --- | --- |
| `autograd` | [draw_goldstein_graph.ipynb](autograd/draw_goldstein_graph.ipynb) | Function evaluation, backpropagation, and computation graphs |
| `autograd` | [draw_sin_higher_order_derivatives.ipynb](autograd/draw_sin_higher_order_derivatives.ipynb) | First through third derivatives of sine |
| `autograd` | [optimize_by_newtons_method.ipynb](autograd/optimize_by_newtons_method.ipynb) | Newton's method with second derivatives |
| `regression` | [regression_simple.ipynb](regression/regression_simple.ipynb) | Manual linear regression training |
| `regression` | [nonlinear_regression.ipynb](regression/nonlinear_regression.ipynb) | Fitting nonlinear data with a two-layer network |
| `neural_network` | [using_model_simple.ipynb](neural_network/using_model_simple.ipynb) | Defining a network with Model and Linear |
| `neural_network` | [iris_lenet_benchmark.ipynb](neural_network/iris_lenet_benchmark.ipynb) | Matched MyDeZero and PyTorch LeNet benchmarks on Iris, training diagnostics, hyperparameter search, and frozen ResNet50 features |
| `neural_network` | [cpu_optimization.ipynb](neural_network/cpu_optimization.ipynb) | CPU weight dtype, inference pooling, spatial and pointwise convolution, and end-to-end ablation against the original paths and PyTorch |
| `neural_network` | [train_mnist_mlp.ipynb](neural_network/train_mnist_mlp.ipynb) | Training an MLP on MNIST |
| `neural_network` | [train_mnist_with_gpu.ipynb](neural_network/train_mnist_with_gpu.ipynb) | Training an MLP on MNIST with GPU acceleration |
| `neural_network` | [weights_saved_loaded.ipynb](neural_network/weights_saved_loaded.ipynb) | Saving and loading trained weights |
| `neural_network` | [pretrained_cnn_validation.ipynb](neural_network/pretrained_cnn_validation.ipynb) | Official-weight comparison with DeZero and CPU/CUDA benchmarks for global average pooling and full ResNet50 inference |

Computation graphs are saved in the project's `draft/` directory.
Notebooks retain their saved outputs; rerun them after code changes.

The Iris benchmark uses scikit-learn and PyTorch. Enable its CUDA comparison
in a kernel with CUDA-enabled PyTorch and CuPy. The transfer learning section
requires MyDeZero's ResNet50 implementation and a compatible local pretrained
archive. The search section runs grid and random search by default. For the
optional TPE comparison, install Optuna in the kernel and enable `RUN_TPE_SEARCH`.

The CPU optimization notebook uses NumPy, CPU PyTorch, scikit-learn, and
threadpoolctl. MNIST uses the standard MyDeZero dataset cache.
