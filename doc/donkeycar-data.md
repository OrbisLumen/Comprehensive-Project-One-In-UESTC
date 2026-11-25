# Seeded DonkeyCar datasets

Follow [the simulator setup](donkeycar-setup.md), activate the project Python environment
and run the commands below from the repository root on Windows or macOS.
Start DonkeySim and use Generated Road. The default connection port is 9091.

## Collect a drive

Choose a seed from [the split manifest](../examples/donkeycar/road-seed-splits.json).
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
