# GPU Optimization Experiment Plan

## Objective and current evidence

Reduce MyDeZero GPU training and inference latency while preserving operator
semantics, gradients, and model quality. The Iris experiment motivates the work.
Small models can spend substantial time in dispatch and short kernels. Larger
CNNs can spend more time expanding convolution windows, moving intermediate
arrays, and computing convolution gradients. Profiling will test these hypotheses.

The `feat/gpu` branch already contains FP32 CUDA Adam fusion and its numerical
tests. Its saved example measured a 6.03x optimizer-update P50 speedup and
1.67–1.83x MLP training-step speedup across batches 1, 16, and 64. These are
historical results for that run. Re-measure them with every final comparison.

This document specifies planned experiments. Proposed performance thresholds
are engineering decision criteria and are subject to agreement before execution.
CPU work is tracked separately in `cpu_optimization.ipynb`.

## Baselines and performance bounds

| Reference | Purpose | Conditions |
| --- | --- | --- |
| Original MyDeZero GPU operators, standard Adam | Measure total progress against the original implementation | Fixed FP32 values, same model, data and update semantics |
| Original GPU operators, fused Adam | Starting point for each new operator experiment | Change only the selected optimization |
| Each candidate alone and combined with Adam | Attribute individual contributions and interactions | Same initialization and measurement protocol |
| All accepted GPU changes | Measure accumulated training and inference benefit | Also run leave-one-out variants |
| Native DeZero GPU | Compare with the source framework | Record checkout revision and verify comparable math first |
| PyTorch eager CUDA | External engineering target | Same architecture, weights, loss reduction, precision and device |
| PyTorch compiled or CUDA Graph replay | Additional optimized reference | Report compilation/capture and execution separately |

Native DeZero and MyDeZero differ in optimizer details and autograd. For a
controlled native-DeZero training comparison, use aligned SGD without momentum
or explicitly align and verify Adam's equations, epsilon placement, and counters.
Keep each framework's default-Adam experiment separately labeled if included.
Initialize from shared arrays to distinguish runtime differences from initializer
rounding. Report the actual default-dtype behavior as an additional comparison.

The original path is the baseline to beat. PyTorch is an empirical target and
can be slower for a particular small workload. Hardware limits are estimates
of the best possible time under stated assumptions, not promised speedups.

Use the following definitions with identical timing scopes.

- Speedup = reference latency / candidate latency. Values below 1 are regressions.
- Latency reduction = 1 - candidate latency / reference latency.
- Throughput = completed samples / elapsed seconds for a timed block of steps.
- PyTorch latency gap = MyDeZero latency / PyTorch latency.
- For an optimized fraction `p` and local speedup `s`, Amdahl's estimate is
  `1 / ((1 - p) + p / s)`. The limiting estimate as `s` grows is `1 / (1 - p)`.
- A kernel's idealized time lower bound is
  `max(required FLOPs / applicable compute ceiling, required bytes / bandwidth ceiling)`.
  Obtain effective ceilings from independent large GEMM and streaming-memory
  measurements where possible. Label theoretical specifications separately.

Launches, synchronization, layout conversion, and imperfect parallelism add time
above those idealized bounds. Determine `p` from representative traces. Overlap
means sums of independently timed phases can exceed the complete step time.

## Shared workloads

### Operator fixtures

| Family | Fixture grid |
| --- | --- |
| Pointwise convolution | `N` in 1, 16, 64, `IC/OC` in 6/16, 64/64, 256/64, spatial sizes 7, 14, 28 |
| Spatial convolution | LeNet 1-to-6 and 6-to-16 with 5x5 filters, MNIST 16-to-32 with 3x3, VGG-style 128-to-256 at 56x56, ResNet-style 64-to-64 at 28x28 |
| Pooling | 2x2 stride 2, odd/even sizes, channels 6, 16, 64, 256, batches 1, 16, 64 |
| Classification loss | Classes 3, 10, 100, 1000 and batches 1, 16, 64, 256 |
| Adam | Equal total parameter counts split into few large versus many small tensors, plus real MLP/CNN parameter lists |

Use a manageable representative grid first, then broaden the winning paths.
Record each complete tensor shape, layout and stride in the results. Include
contiguous NCHW, channels-last storage with NCHW logical dimensions, sliced and
reversed arrays in correctness checks. Benchmark supported fast layouts and
their conversion costs. Include the generic path's unsupported-case behavior.

### Complete models and datasets

| Scale | Dataset and model | Purpose |
| --- | --- | --- |
| Small | Synthetic MLP and Toy CNN | Dispatch and short-kernel sensitivity |
| Small CNN | Stratified Iris split, 120 train / 30 held out, LeNet-style encoding | Reproduce the motivating workload |
| Medium CNN | MNIST, balanced 1,200 train / 500 held out, configurable-width CNN | Repeatable training and operator interactions |
| Larger CNN | CIFAR-10 and an explicitly specified residual classifier | Test realistic 32x32 image training after the first screening |
| Application inference | ResNet50 and VGG16, fixed images and compatible shared weights | Check end-to-end inference relevance |

Use batches 1, 16, and 64 for the core suite. Add 128 or 256 for MLP/MNIST when
memory permits. Start 224x224 application inference at batches 1, 4, and 8.
Record OOM configurations and compare the largest batch feasible for every
implementation. Reserve final held-out data for evaluation. Standardization
and hyperparameter selection use training/validation data only.

Synthetic operator fixtures do not establish dataset quality or full-model
speed. Use seeded random weights for initial inference screening. Shared
pretrained checkpoints provide the final application-level check when available.

## Experiments and feasibility

### E0. Diagnose the baseline

Question: which fraction of a step is dispatch, transfer, convolution, pooling,
loss, backward, or Adam, and how does it change with batch size?

Instrument forward, loss, backward, update, input preparation, and transfer
ranges. Within convolution, distinguish window extraction, contraction/layout
work, input gradients, and weight gradients. Use resident data first, followed
by a complete input-pipeline run.

Collect synchronized wall latency, CUDA Event elapsed time, CPU submission time,
kernel launch counts, timeline gaps, allocation behavior, and transfer volume.
Use Nsight Systems for launch/timeline diagnosis and selected Nsight Compute
runs for kernel efficiency. Profiler overhead is excluded from headline timing.
Document missing counter permissions or unsupported tools as unavailable metrics.
E0 produces a ranked bottleneck table and an Amdahl estimate for each candidate.

### E1. Establish fused Adam as an independent contribution

Question: how much update work and complete training time does fusion remove?

Compare standard Adam and the existing one-kernel-per-tensor FP32 update using
identical gradients, moments, counts and hyperparameters. Repeat across parameter
sizes, tensor counts and actual model parameter lists. Include missing gradients,
hooks, noncontiguous arrays, overlapping buffers, FP64 and mixed-dtype fallback.

Observe update latency, kernels per update, effective memory traffic, and update
share of complete training time. Check weights and both moments over multiple
steps, including tiny gradients and gradients near zero. Existing rounding and
fallback tests are part of the acceptance checks. This experiment is already
implemented for one MLP and will be expanded to the common workload matrix.

### E2. Bypass expansion in pointwise convolution

Question: can 1x1 convolution avoid `im2col` and extra contraction packing?

For stride 1 and zero padding, compare the original path with direct cuBLAS-backed
matrix multiplication for forward, `dX`, and `dW`. Measure batched and flattened
matrix formulations with their required layout conversions. Preserve higher-order
gradients through differentiable backward paths or use the generic path when
recording a higher-order graph.

Observe phase latency, temporary bytes, copy volume and ResNet-block training
time. This has a limited algorithmic scope and reuses existing matrix libraries.
Choose supported layouts from measurements. Include conversion in the reported
cost and retain slower shapes in the experiment output.

### E3. Implement direct GPU pooling kernels

Question: can direct window reads and gradient routing remove pooling's expanded
buffers without changing its selection rules?

Start with unpadded 2x2 stride-2 pooling. Use CUDA C/C++ through CuPy to compute
the output directly. Under gradient recording, store compact winning indices.
For nonoverlapping windows, write backward gradients directly to the input
gradient buffer with all nonselected positions initialized correctly. Evaluate
general overlapping windows separately because gradient accumulation changes
the parallel algorithm and may introduce atomic operations.

Observe forward/backward latency, allocation peaks, launch count and memory
traffic. Verify first-max tie handling, signed zeros, NaNs, infinities, odd sizes,
noncontiguous input, and second derivatives against current behavior. The
inference-only variant and training variant receive separate results.

### E4. Fuse classification loss and its first derivative

Question: do stable row reductions and fewer intermediate arrays improve loss
and backward time at small and large class counts?

Compare the existing softmax-cross-entropy path with stable log-sum-exp reduction
and a fused gradient calculation using CUDA kernels. Preserve averaging over the
batch, label formats, output dtype, upstream scaling and supported higher-order
gradients. Keep the differentiable generic backward available where graph
recording requires it. Choose block/warp strategies by the class-count regime.

Observe loss/gradient latency, launches, temporary memory and complete MLP/CNN
step time. Test extreme logits, underflow, batch 1, one class, all supported
label shapes, and multiple upstream gradient values. This requires more
reduction and numerical work than Adam, so benchmark feasibility before promotion.

### E5. Compare convolution strategies and cuDNN

Question: how much CNN time can be saved by removing explicit expanded windows
and using optimized convolution implementations?

Run three operator strategies on the same FP32 data where supported.

1. Current explicit expansion plus contraction.
2. A CUDA window-packing kernel that writes directly to the matrix layout needed
   by GEMM, including a matching input-gradient accumulation strategy.
3. An optional cuDNN path for forward, backward-data and backward-filter.

Measure each direction separately before complete training. Include input/output
reorders, weight transforms, execution-plan preparation, scratch space, and plan
reuse in the relevant timing scope. Plan caching keys must include shapes,
strides, dtype, convolution parameters, device and math mode. Repacked weight
data must be refreshed after parameter updates. Per-call scratch space must be
safe under concurrent streams. Test changing shapes and repeated calls.

Prioritize a cuDNN FP32 prototype after E0 confirms a convolution bottleneck.
Implement direct packing only when traces identify avoidable layout traffic or
when it provides a useful dependency-free path. A generic hand-written GEMM or
Tensor Core convolution is a separate, higher-cost research project.

Compare strict FP32 first, explicitly recording TF32 settings and selected
algorithms. Test deterministic requirements and numerical differences. Preserve
the generic implementation for unsupported inputs and higher-order gradients.
Choose performance dispatch only after measuring realistic shapes. The gain
from a bare cuDNN call is insufficient to establish a framework-level benefit.

### E6. Extend Adam fusion across parameter tensors

Question: after E1, do many small parameter tensors still create launch overhead?

Prototype an update kernel that processes multiple eligible tensors per launch.
Compare standard Adam, existing per-tensor fusion and the multi-tensor candidate
at equal total element counts. Include packing pointer/metadata lists and
rebuilding them after parameters change in end-to-end measurements.

Observe launch count, CPU submission time and training latency. Check per-tensor
counts, skipped parameters, hooks, aliasing, devices, and moment state. Promote
this work when E0/E1 show an appreciable optimizer dispatch contribution.

### E7. Reduce repeated dispatch with CUDA Graphs

Question: after operator optimization, do host gaps still dominate fixed-shape
execution?

Start with fixed-shape inference and resident buffers. Compare eager execution,
captured replay, and eager PyTorch. Measure capture/instantiation separately,
replay latency, retained memory, and the number of replays needed to amortize
setup. Copy new input values into stable buffers and verify changed outputs.

Training capture needs a feasibility prototype. Python-side Adam counters and
bias-correction scalars evaluated during capture can become fixed on replay.
Device-side state, gradient clearing, stable parameter/moment storage and
allocation lifetime must be correct over successive replays. Compare many
updates with eager execution before measuring performance. Include changing
inputs and supported shape changes. Use an eager path for cases outside capture.

### E8. Evaluate the input pipeline

Question: do faster kernels leave data preparation or host-to-device transfer
as the next bottleneck?

Compare already resident batches, per-batch synchronous transfer, and a prototype
using pinned host buffers and asynchronous prefetch with explicit stream/event
dependencies. Account for the lifetime and reuse of buffers. Keep dependencies
correct across loss, update, and next-batch consumption.

Observe total epoch time, transfer time/volume, timeline overlap and completed
samples per second. Small resident datasets and streamed image datasets require
separate interpretations. Promote pipeline changes only if complete workloads
show a bottleneck and a repeatable gain.

## Timing and hardware metrics

Record the commit, GPU model, driver, CUDA runtime, CuPy, cuDNN, PyTorch, NumPy,
OS, power mode, precision policy, stream use and memory-pool state for each run.
Benchmark one variant at a time without other GPU jobs. Note throttling or power
changes. Record actual software versions because the source references may
describe newer versions than the installed environment.

| Metric | Measurement and interpretation |
| --- | --- |
| Synchronized wall P50/P95 | Complete function or step latency, including host dispatch and completion |
| CUDA Event elapsed P50/P95 | Time between events on the measured stream, including device-side gaps between them |
| CPU submission time | Host time to enqueue work, distinct from waiting for completion |
| Block throughput | Synchronize around a block of completed steps to reduce per-call instrumentation effects |
| Kernel count and launch gaps | Timeline explanation of dispatch and fusion effects |
| Device memory | Live tensor bytes, extra workspace and retained pool bytes reported separately |
| Allocation peak | Use an appropriate allocator hook or profiler and account for reusable pools |
| Effective bandwidth | Explicitly defined transferred bytes divided by elapsed time |
| Compute throughput | Count FLOPs consistently, counting a multiply-add as two operations |
| SM activity, occupancy, cache and DRAM metrics | Selected profiler counters to explain a kernel result |
| GPU utilization and power | Supporting observations over sufficiently long runs |

For convolution forward, the nominal count is
`2 * N * OH * OW * OC * IC * KH * KW` FLOPs for an ungrouped convolution.
Report backward operations separately and document conventions. Adam is normally
better explained with memory traffic and launches than nominal FLOPs.
High utilization or occupancy alone does not establish better performance.
CuPy pool-reserved bytes and PyTorch reserved bytes represent different allocator
states. Use matched live-data scopes when comparing memory across frameworks.

For timing, prewarm each shape and implementation until compilation and plan
selection finish. Start with 7 rounds and at least 20 samples per variant per
round, rotate order, and repeat in 3 independent runs. For sub-millisecond work,
also time blocks. Retain individual samples. Use paired round summaries and
bootstrap confidence intervals for speedup, keeping the run/round grouping.
Increase repeats if the effect is comparable to timing variation.

Separately report first-use context/compile/plan/capture latency and warmed
execution. For transfer experiments include transfers explicitly. Synchronize
all participating streams before stopping a complete wall-clock measurement.
Avoid scalar reads or logging inside the timed step unless they are intentionally
part of the workflow. Profile separately from latency measurement.

## Numerical and training quality

First compare forward outputs and random-vector gradient products against the
existing path. Record maximum absolute error and relative L2 error, including
the zero-reference case. Use FP64 finite differences on small suitable inputs.
Use operation-specific tolerances established before comparing candidates.
Check parameter and optimizer-state agreement over successive updates.

For framework comparisons, validate labels, loss reduction, ReLU/pooling boundary
rules, gradients and optimizer math. Use an aligned optimizer for controlled
training comparisons. Report any default-framework comparison separately.

Train 5 paired seeds for the final Toy/Iris/MNIST quality runs. Set the same
initial arrays, batches, data order, epoch budget and optimizer settings.
Freeze these settings before evaluating final held-out quality.

- Report accuracy, macro precision, macro recall, macro F1 and held-out loss.
- Plot training/validation loss and accuracy, gradient norms and update norms.
- Report NaN/Inf counts, divergence, and convergence variability.
- Compare a predeclared loss/accuracy target, epochs and wall time to target.
  Mark a target not reached within the budget explicitly.
- Show paired score differences and uncertainty across seeds. A proposed
  noninferiority margin is 1 percentage point in accuracy and macro F1 on
  sufficiently sized held-out sets. Choose a task-appropriate margin before
  running the final comparison.

Iris has 30 held-out examples, so one classification change is 3.33 percentage
points. Report exact changed predictions and seed variability. This set alone
cannot resolve a one-point quality claim. FP32 rounding can change long training
trajectories, so unchanged final labels from one checkpoint are only one check.
Mixed precision, TF32 acceleration and loss scaling would receive a separate
experiment with their own numerical policy if added later.

## Ablation and promotion decisions

For every candidate `X`, compare original, Adam only, `X` only, and Adam plus `X`.
For inference, report the actual inference-relevant variants, since Adam does
not execute there. Once multiple changes are accepted, report the cumulative
sequence and all-accepted-minus-one configurations.

Use a full factorial experiment for the first three independent operator changes
when practical. Treat mutually exclusive convolution backends as alternatives.
Measure additional pairwise interactions where a shared layout or allocation
changes the result. Include E6 against E1 and E7 against the best eager path.

Promote candidates according to the following proposed criteria.

1. Correctness, state consistency, fallback and relevant higher-order tests pass.
2. The measured target region improves beyond run-to-run uncertainty. Use 10%
   warmed operator P50 improvement as an initial engineering target.
3. A full workload benefits or the isolated operator has a demonstrated target
   use case. Use 5% complete-step/inference improvement as an initial target.
4. Examine P95, workspace, first-use cost and quality alongside P50. Repeated
   regressions above 5% in an important region require measured dispatch choices
   or further work before enabling that path by default.
5. The paired training comparison supports the predeclared quality criterion.

Thresholds guide investment and default dispatch. They do not justify discarding
slow measurements. Uncertain or failed candidates remain visible with their
tested shapes, causes, and next decision. Do not claim a general speedup from
the best single fixture.

## Notebook and evidence layout

Keep `examples/neural_network/gpu_optimization.ipynb` as the main GPU entry point.
Expand it in the following order as real experiments become available.

1. Problem, environment, reproducibility settings and baseline definitions.
2. E0 phase breakdown and baseline latency/throughput by workload and batch.
3. E1 existing Adam fusion, followed by its expanded model coverage.
4. E2–E6 individual candidate sections with hypotheses, feasibility and outcomes.
5. E7 graph and E8 pipeline sections when their prerequisites are established.
6. Cumulative and leave-one-out ablation, matched PyTorch/native DeZero tables.
7. Paired training quality, limitations, regressions and conclusions by regime.

Each candidate section contains the problem, predicted source of savings,
implementation strategy, supported region, correctness checks, timing table,
figure, model impact and acceptance decision. Use explicit run switches for
expensive or optional-backend experiments and display a clear skipped status.
Do not place prospective numbers in executed result cells.

Required figures are phase bars, latency/throughput versus batch, individual
operator speedups, allocation/workspace comparisons, ablation heatmaps, paired
quality differences and convergence curves. Include errors or intervals and the
baseline at speedup 1. Keep raw P50/P95 and shape tables alongside the figures.

If notebook size or optional tooling makes a section cumbersome, extract that
single executable experiment into a clearly linked GPU notebook and leave its
summary in the main entry point. Benchmark helpers, if shared, belong alongside
the examples and must not introduce runtime dependencies into MyDeZero.

Keep full individual timings, machine-readable metadata and profiler traces
under an ignored `draft/gpu-benchmarks/<run-id>/` directory. Commit concise tables,
actual outputs, figures and reproduction code. Optional compact JSON result
artifacts should be justified by reproducibility needs. Retain no personal paths
in shared setup instructions. Standard MyDeZero dataset caching handles data.

Correctness tests belong under `tests/mydezero/`. New tests target numerical
behavior, boundary semantics and state lifetime. Performance comparisons live
in examples. Update `api.md` for supported dispatch, `testing.md` for coverage,
and the examples index when a candidate becomes implemented.

## Execution order

### Initial implementation results

The notebook retains the historical three-stage screening results. The framework
now includes pointwise GPU forward/input/weight gradients, direct 2x2 GPU pooling,
and FP32 CUDA cross entropy, alongside the existing fused Adam.

Across the nine Toy/Iris/MNIST model and batch settings, warmed all-on inference
improved 1.26–1.40x and training steps improved 1.93–2.15x. Optimized training
latency was 1.03–1.22x PyTorch eager in this run. Five paired seeds gave matching
Toy/Iris mean quality and MNIST accuracy of 92.60% versus 92.68%.

Official-weight VGG16 inference improved about 5–6% at batches 1 and 4. ResNet50
inference regressed about 2–4%. These results limit general CNN speedup claims.
The cuDNN algorithm-0 feasibility probe improved individual convolution
directions by 1.15–2.47x and still needs portable backend and full-model acceptance.

The equal-size multi-tensor Adam prototype, graph replay and prepared pinned
transfer remain examples. CuPy 13.6 rejected full CNN graph capture at cuBLAS,
so the replay experiment measures pooling plus ReLU. Captured training and
complete-epoch prefetch remain pending. Compatible Nsight counters were not
collected. Timing repeats were three blocks within one process. Independent
process replication remains a final acceptance follow-up.

The regression suite passed 313 tests on Windows with one CUDA device. macOS
was unavailable. Native DeZero's MLP comparison used aligned SGD and a temporary
import compatibility alias for its removed NumPy `int` name.

### Large-model regression follow-up

The follow-up replaces the initial large-model acceptance conclusions above.
It identifies three separate problems and retains controls for each correction.

- Batched 1x1 GEMM regressed at some late ResNet shapes. Flattening batch and
  spatial rows into one GEMM improves forward and input-gradient execution.
- Integer count arrays widened ResNet global-pooling outputs and BatchNorm
  gradients to FP64. Counts and epsilon use at least FP32 precision, including
  gradient-graph recording. The FP32 reference isolates this correction while
  retaining original operators and standard Adam.
- Official VGG dense weights are Fortran-contiguous. The earlier C-only Adam
  eligibility rule excluded its largest tensors. Fusion now supports that
  layout using packed gradients and flat physical views. Timings include packing.

The initial cuDNN probe also used the allow-conversion math constant instead of
FMA math. Its historical timings cannot establish strict-FP32 acceleration.
The corrected optional backend requests FMA, restricts algorithm families and
workspace, caches descriptors, and refreshes weight values on every call.
The new complete-model comparisons explicitly disable TF32.

`gpu_large_workloads.py` measures official-weight ResNet50 at batches 1/4/16 and
VGG16 at batches 1/4/8 on resident synthetic FP32 224x224 inputs. Inference
alternates variants on one checkpoint. Training reloads the checkpoint per
variant, seeds MyDeZero dropout equally, and rotates variant order across three
fresh processes. Each process warms three calls, measures inference in five
rounds of five calls, and training in three rounds of three calls. This smaller
large-model timing budget differs from the core screening protocol. Retain raw
samples and report the three-process range without calling it a confidence
interval. Training is a complete forward/loss/backward/Adam step.

The following are medians of process P50 latencies in milliseconds. Default
means FP32 corrections, flat pointwise GEMM, pooling/loss kernels and layout-aware
fused Adam. Optimizer-only variants reuse the same paired inference measurement.

| Model and batch | Original inference | Default inference | cuDNN inference | Original training | Default training | cuDNN plus layout-aware Adam training |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50, 1 | 11.76 | 10.46 | 10.71 | 244.60 | 42.87 | Rejected |
| ResNet50, 4 | 23.99 | 21.80 | 19.32 | 422.64 | 87.24 | 77.12 |
| ResNet50, 16 | 64.43 | 61.66 | 50.18 | 1386.32 | 243.56 | Rejected |
| VGG16, 1 | 13.17 | 12.41 | 11.13 | 196.88 | 54.09 | 48.77 |
| VGG16, 4 | 41.77 | 39.38 | 23.44 | 272.89 | 148.01 | 106.15 |
| VGG16, 8 | 81.08 | 76.46 | 42.73 | 367.95 | 232.23 | 165.83 |

At ResNet50 batch 16, the FP32-only reference takes 305.53 ms per training step.
The default path adds a 1.25x gain over that corrected reference. Its 5.69x
paired speedup over original behavior includes removal of unintended FP64 work.
For VGG16 batch 8, C-only fusion takes 354.61 ms, layout-aware fusion takes
232.23 ms, and experimental cuDNN plus layout-aware fusion takes 165.83 ms.
Matched PyTorch eager training takes 106.61 ms for ResNet50 batch 16 and
159.28 ms for VGG16 batch 8. ResNet50 retains a substantial engineering gap.

Logits pass tolerance and prediction agreement checks. A first-step gradient
relative-L2 gate of 0.005 against the corrected FP32 reference rejects cuDNN
ResNet50 training at batches 1 and 16, with errors 0.008288 and 0.005555.
Those configurations receive no training speedup claim. Batch 4 and all tested
VGG configurations pass this gate. cuDNN remains experimental and opt-in through
`using_config('use_cudnn', True)`, with the default disabled.

The regression suite passes 326 tests, including second derivatives, changed
weights, nondefault streams, FP64 fallback, FP32 preservation and Fortran Adam
state agreement. Counts above the FP16 finite range also receive regression
coverage. These resident synthetic workloads establish compute latency
and first-step checks. They do not establish convergence or held-out accuracy
for large datasets. CIFAR quality, complete input-pipeline timing, cold-start
cost and profiler-counter coverage remain separate acceptance work.

### Three-stage implementation

| Stage | Experiments | Deliverable and gate |
| --- | --- | --- |
| 1. Baseline and direct operators | E0, E1, E2, E3 | Phase timings, Adam controls, pointwise forward/input/weight gradients and direct pooling, followed by numerical and higher-order tests |
| 2. Loss and convolution | E4, E5 | Fused loss with differentiable higher-order fallback, and a cuDNN feasibility probe that includes descriptor setup and warmed calls |
| 3. Scheduling and acceptance | E6, E7, E8 and combined ablation | Restricted multi-tensor update screen, changed-input graph replay, prepared-transfer comparison, paired training and matched-framework timing |

The first implementation pass uses the Toy/Iris/MNIST suite. The large-model
follow-up adds complete synthetic ResNet50/VGG16 steps and an optional cuDNN
backend. Large-dataset quality, macOS packaging, captured training and
complete-epoch prefetch remain conditional follow-up work.
The executed GPU notebook distinguishes framework paths from example prototypes.

Complete the CPU PR review separately. Continue profiling and isolated GPU
experiments on `feat/gpu`. Once the CPU PR merges, sync its dtype fix and other
shared behavior before final GPU acceptance measurements.

Run E0 and expand E1 first, then E2 and E3. E4 follows if loss/dispatch is material.
Prioritize E5 when convolution dominates. E6 and E7 depend on the remaining
launch overhead. E8 follows observed pipeline limits. Preserve one reviewable
implementation commit per accepted change, followed by tests and documentation.
Create follow-up task branches from the current main branch when the work is
split into later PRs. Profiling can change the order, with evidence recorded.

## Technical references

- CuPy recommends asynchronous-aware timing, warm-up and attention to first-use
  costs. [Performance best practices](https://docs.cupy.dev/en/stable/user_guide/performance.html)
- CuPy supports elementwise, reduction and raw CUDA kernels for different levels
  of control. [User-defined kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html)
- cuDNN execution involves device data and managed handles, with explicit context
  and stream considerations. [Core concepts](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html)
- Focused timeline collection helps examine selected execution phases.
  [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- Kernel counters and collection modes support detailed kernel analysis.
  [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
