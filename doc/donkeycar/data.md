# DonkeyCar datasets

Follow [the simulator setup](setup.md), activate the project Python environment
and run the commands below from the repository root on Windows or macOS.
Start DonkeySim and use Generated Road. The default connection port is 9091.

## Choose a recording workflow

| Purpose | Entry point | Output |
| --- | --- | --- |
| Record keyboard driving | `python -m donkeycar_app.record` | A session under `data/raw/donkeycar/` with `samples.csv` (image, steering, throttle) and PNG frames in `images/`. |
| Train on seeded PID drives | Run `collect_seeded.py` and then `process_seeded.py` as shown below. | The processor reads native `record_*.json` files and JPEG frames from each seeded archive and writes split indices for the notebooks. |

## Collect a seeded drive

Choose a seed from [the split manifest](../../examples/donkeycar/road-seed-splits.json).
To verify a layout and leave it open for inspection, run:

```console
python src/donkeycar_app/collect_seeded.py --seed 8345373 --check-only --preview
```

The script compares sampled road nodes across two seeds and regenerates the
requested seed to verify repeatability. Press Enter to finish the preview.
If verification requires a fresh simulator, use `--sim-exe PATH` to specify
the executable for your installation.

For collection, select a fresh simulator log directory, manually enter Generated
Road and confirm that **Auto Drive w Rec** is visible. Pass the same directory
to the script:

```console
python src/donkeycar_app/collect_seeded.py --seed 8345373 --log-dir data/raw/seed-8345373
```

Follow the terminal prompts to start and stop recording in the simulator.
The script checks the layout again and copies the recording into
`data/raw/donkeycar-seeded/` with its seed, sampled nodes and layout hash in
`collection.json`. To identify a road already open in the simulator, run
`python src/donkeycar_app/collect_seeded.py --show-seed`. This matches the
current nodes to archived recordings. An unseen layout returns an unknown seed.

## Process the splits

After collecting all seeds in the manifest, run:

```console
python src/donkeycar_app/process_seeded.py
```

The script writes `train.json`, `validation.json` and `test.json` under
`data/processed/donkeycar/seeded/`. Image and record paths are relative to the
repository root. Keep the archived recordings alongside the indices.
Rerunning the command refreshes the indices. Processing checks frame counts
and rejects repeated seeds, sampled layout hashes shared across splits and
byte-identical images shared across splits.

Each row retains the original steering command in `raw`. Training rows with a
full-lock command opposite to both immediate neighbors receive weight 0.1
when frame IDs are consecutive and throttle is positive across all three
frames. For isolated spikes, `target` is the mean of the neighboring commands.
Validation and test rows retain the original target and unit weight.

Use these indices with the [global 2048 notebook](../../examples/donkeycar/resnet50_single_frame_global_2048.ipynb)
or the [spatial 9728 notebook](../../examples/donkeycar/resnet50_single_frame_spatial_9728.ipynb).
Each notebook opens with a table showing which label and loss weight each recipe uses.

## Compare steering recipes and use the checkpoints

Both notebooks cross three treatments of flagged spikes with three weights for
large non-spike commands (absolute raw steering at least 0.3).

| Training label | Spike weight | Large non-spike weight 1 | Weight 2 | Weight 4 |
| --- | ---: | --- | --- | --- |
| Original command | 1 | `raw` | `raw_x2` | `raw_x4` |
| Original command | 0.1 | `downweighted_x1` | `downweighted_x2` | `downweighted_x4` |
| Interpolated target | 1 | `smoothed_x1` | `smoothed_x2` | `smoothed_x4` |

A completed run writes ten full-model checkpoints in
`models/donkeycar/<run-id>/`. Each recipe's `.npz` saves its epoch with the
lowest overall validation MAE against original PID commands. `best.npz` copies
the winning recipe's checkpoint. Ties retain the first minimum encountered.

The matching `runs/donkeycar/<run-id>/metrics.json` identifies the selected
recipe and epoch. `experiment.png` shows a local steering trace and the nine
validation curves. The global 2048 notebook writes `validation-prediction.csv`
and `test-prediction.csv` for the winner, with one row per frame and columns
`route`, `frame`, `target` (original PID command), `prediction`, and `recipe`.
The spatial 9728 notebook uses `validation-predictions.csv` and
`test-predictions.csv` with the same first four columns.
