# Course CNN steering baseline

AutoDriveNet follows the DonkeyCar PyTorch handout (pp. 11–15). It converts 120 × 160 RGB frames to OpenCV HSV, scales all three channels by 1/255, trains five convolutional layers end to end, and flattens the spatial features into the steering head. Training uses original simulator steering, MSE, and Adam at 0.0001. The simulator clips the model command to the interval [-1, 1] through the driving entry point.

## Reproduce

Use the project environment and the simulator setup in [DonkeyCar setup](donkeycar-setup.md). Run the following on Windows or macOS with the environment activated. The training command expects each native drive under `data/raw/trace1` through `data/raw/trace8`. Supply `--root` if the data is in another checkout. Choose a new output directory for each run.

```shell
python -m pip install -e ".[autodrive]"
python -m donkeycar_app.train_autodrivenet --out runs/donkeycar/course-baseline
```

Fit on trace1–5 (15,857 frames), select a checkpoint by validation MSE on trace6 (3,246), then evaluate once on the historical holdout trace7–8 (7,810). All eight drives use the simulator's built-in driver. Evaluation CSVs store clipped commands in original frame order. The epoch log records validation loss before command clipping. The reference run used seed 42, batch size 100, a 24-epoch limit and early stopping after five epochs without improvement. Epoch 24 had the lowest validation MSE within this budget.

To drive with the CNN alone, start the simulator at its menu and run:

```shell
python -m donkeycar_app.drive_autodrivenet --checkpoint runs/donkeycar/course-baseline/best.pt --out runs/donkeycar/course-baseline/drive.csv
```

The paired comparison also needs the prior SpatialSteeringResNet source checkout, its MyDeZero dependencies, and its saved weights. Both controllers run at throttle 0.20. They alternate order across three pairs, reset the vehicle between episodes, and share one loaded Generated Road scene. The logged path length sums horizontal position changes in simulator units. The run ends on a lap, environment termination, or the 3,000-step limit.

```shell
python examples/donkeycar/compare_steering.py --course runs/donkeycar/course-baseline/best.pt --resnet path/to/resnet/best.npz --resnet-src path/to/resnet/src --out runs/donkeycar/course-baseline/paired
python examples/donkeycar/plot_steering.py runs/donkeycar/course-baseline/validation-predictions.csv path/to/resnet/validation-predictions.csv runs/donkeycar/course-baseline/validation-bends.png
```

## Results

The comparison uses the existing spatial ResNet run `20261123T145241Z-ee0bda`. Its frozen backbone, linear readout and downweighted training differ from the course CNN recipe. The comparison measures the two complete recipes.

| Metric | AutoDriveNet | Spatial ResNet |
| --- | ---: | ---: |
| Validation MAE | 0.0826 | 0.1026 |
| Test MAE | 0.0721 | 0.0942 |
| Test MAE on 949 frames with absolute steering ≥ 0.3 | 0.1780 | 0.4328 |
| Test MAE on 708 of those frames without flagged spikes | 0.161 | 0.262 |
| Test MAE on 241 flagged spikes | 0.227 | 0.935 |
| Completed laps in three starts | 0 | 0 |

The spike subset uses the existing ResNet evaluation rule, which flags a full-lock command flanked by opposite-sign commands on consecutive moving frames. It describes the label pattern. It does not establish whether a command was necessary. The curve plot selects the strongest positive and negative 21-frame mean expert commands on trace6 and shows 30 frames on each side of each center. In the left bend around frame 2256, the course CNN reproduced several opposite-direction peaks and added others. In the right bend around frame 2633, it reproduced some reversals with smaller magnitudes.

| Pair | CNN steps | ResNet steps | CNN horizontal path length | ResNet horizontal path length |
| --- | ---: | ---: | ---: | ---: |
| 1 | 126 | 287 | 6.80 | 23.62 |
| 2 | 128 | 181 | 7.03 | 12.23 |
| 3 | 124 | 181 | 6.66 | 12.24 |

All six episodes ended with absolute cross-track error above 8 and no reported collision. AutoDriveNet had lower held-out imitation error and left the road earlier in every pair. The three repetitions cover one scene. Path length measures ground covered, including lateral motion. The next diagnostic is to inspect the camera, applied command, and signed cross-track error just before the repeated CNN departures, and check the native recorder's image/action timing against the external control loop.
