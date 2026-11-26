# DonkeyCar Setup

MyDeZero and vehicle control share a root `.venv`. Each member creates their own environment.
Use the same Python and dependency versions on Windows and macOS, with the simulator for each platform.

## Files and Versions

| Item | Location |
| --- | --- |
| Third-party interface source | `tools/gym-donkeycar/` |
| Simulator | `tools/donkey-sim/` |
| Manual driving recorder | `src/donkeycar_app/record.py` |
| Raw driving data | `data/raw/donkeycar/` (ignored by Git) |
| Requirements and templates | `request/` |
| Software engineering reports and figures | [MyDeZero UML](../uml/README.md) and `work/software-engineering/` (planned reports) |

Do not commit `tools/` or `.venv`. Store data, weights, and training outputs in ignored
`data/`, `models/`, and `runs/` directories or shared storage. Record download links, checksums, and related
experiments in the repository. Avoid personal absolute simulator paths.

Shared baseline:

- [Interface source](https://github.com/tawnkramer/gym-donkeycar): `a1f4ca6961e17a6929c1f2e883358a1edafb479d`
- Simulator: `v25.10.06`; download the [macOS package](https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimMac.zip) or [Windows package](https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimWin.zip).
- macOS archive SHA-256: `edc36077126257320d433a7d07266c1feae289245687c4b9139bac353e31dce0`
- Windows archive SHA-256: `75bad63a38aa6467b35d99ac4fe0590858b35c2a454ab0a13ffb008e092180a5`
- Python: 3.12. Direct dependencies are pinned in [requirements.txt](../../requirements.txt).
  Transitive dependencies are resolved by pip; this is not a complete lockfile.

Record version changes through PRs. Do not independently track the latest `master`.

## Installation

Run from the repository root. Skip creation or cloning if the files already exist:

```bash
mkdir tools
git clone https://github.com/tawnkramer/gym-donkeycar.git tools/gym-donkeycar
git -C tools/gym-donkeycar checkout a1f4ca6961e17a6929c1f2e883358a1edafb479d
```

Extract the simulator into `tools/donkey-sim/`. On Windows, keep the executable,
data folders, and DLLs in their original relative locations.
Create `.venv` as described in the [root README](../../README.md), then install the dependencies and both projects:

```bash
# macOS
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import mydezero, gymnasium, gym_donkeycar; print('import OK')"
```

```powershell
# Windows PowerShell; activation is optional
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import mydezero, gymnasium, gym_donkeycar; print('import OK')"
```

Use a Python 3.12 interpreter to create the environment (`python --version`).
If `python` opens the Microsoft Store, install Python 3.12 and reopen the terminal.
Do not copy `.venv` between machines. Activation is optional when using its Python path.

Download and extract the Windows simulator from PowerShell:

```powershell
Invoke-WebRequest https://github.com/tawnkramer/gym-donkeycar/releases/download/v25.10.06/DonkeySimWin.zip -OutFile tools/DonkeySimWin.zip
Get-FileHash tools/DonkeySimWin.zip -Algorithm SHA256
# Compare with the Windows checksum above before extracting.
Expand-Archive tools/DonkeySimWin.zip -DestinationPath tools/donkey-sim
```

The executable is `tools/donkey-sim/DonkeySimWin/donkey_sim.exe`.
Keep the archive until its checksum has been verified. For macOS, compare
`shasum -a 256` output with the macOS checksum above before extraction.

## Python and Notebook Checks

Run from the repository root, using `.venv/bin/python` on macOS:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m jupyter --version
.\.venv\Scripts\python.exe -m jupyter lab
```

Select the repository `.venv` interpreter in your editor. If a named notebook
kernel is needed, register it explicitly:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name cnn-drive --display-name "CNN Drive (venv)"
```

Graphviz is optional: install its system executable and add `dot` to `PATH` for
computation-graph PNG rendering. Its integration test skips when `dot` is absent.
GPU/CuPy and other training frameworks are outside this baseline.

On macOS, locate the extracted app with `find tools/donkey-sim -name donkey_sim.app`.
Use the returned path in the commands below. If execute permission is missing, restore it before launching:

```bash
sim_app="tools/donkey-sim/donkey_sim.app"  # Adjust for the archive layout.
chmod u+x "$sim_app/Contents/MacOS/donkey_sim"
open "$sim_app"
```

## Collect Expert Drives on Generated Road

Project I uses the simulator's `generated_road` scene, registered in Gymnasium
as `donkey-generated-roads-v0`.

1. Select `generated_road` in the simulator. Start `Auto Drive w Rec` to let the
   built-in PID controller follow the road and record its camera and controls.
2. Click `Stop` after the drive. Return to the main menu and use `Log dir` to
   locate the export. Keep each drive in its own directory under
   `data/raw/donkeycar/` and preserve the simulator's original files.
3. Check a short export before bulk collection. Verify that image paths resolve
   and that the steering labels cover left turns, right turns, and straight
   segments. Record the road seed if available and split by drive or road seed.

The [simulator recording guide](https://github.com/tawnkramer/donkey/blob/master/docs/guide/simulator.md#recorded-data)
describes the native DonkeyCar tub format. In its per-frame JSON format,
`cam/image_array` points to the frame and `user/angle` and `user/throttle` hold
the expert controls. Use the native recording for supervised training and use
the trained MyDeZero model for closed-loop evaluation.

## Record a Manual Drive

Start the simulator and run the recorder from the repository root. The keyboard
controls and camera preview require a Python installation with Tkinter support.

```bash
# macOS
.venv/bin/python -m donkeycar_app.record
```

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m donkeycar_app.record
```

Click the recorder window. Hold the up arrow to drive at fixed throttle and
hold the left or right arrow to increase steering in that direction. Release
the steering keys to center. Release the up arrow to cut throttle and pause
recording. Press Escape to finish. Start the command again for a new drive.
The default is `donkey-generated-roads-v0` at `127.0.0.1:9091`. Use `--help`
for options, including `--throttle` to adjust the driving speed.

Each drive gets its own directory under `data/raw/donkeycar/`. `images/` holds
the original RGB PNGs, `samples.csv` pairs relative image paths with continuous
`steering` and `throttle` values, and `metadata.json` records the track and
capture settings. Each sample uses the camera frame displayed before its action
was sent. Use the directory name as the run ID when splitting data for training.
Use only the listed images in `samples.csv` and skip drives with no sample rows.

## Drive with the trained ResNet50

Start DonkeySim at the main menu and open the notebook matching your trained head:
[spatial 9728](../../examples/donkeycar/resnet50_single_frame_spatial_9728.ipynb) or
[global 2048](../../examples/donkeycar/resnet50_single_frame_global_2048.ipynb).
Run its model definition cell and enable **Drive a saved checkpoint** immediately below it.
Choose a listed checkpoint by number, or enter an absolute path or a path relative to the kernel working directory.

The shared `run_drive(model.steer)` connects to Generated Road at localhost port
9091, applies model steering at throttle 0.15, and logs each step under
`runs/donkeycar/closed-loop` at the project root. Training run and weight directories start with the notebook name.
Closed-loop CSV names include the notebook name, steering multiplier, recipe and run ID.
For renamed checkpoints, pass the original recipe to `load_steering_model(..., recipe="...")`. Pass
`log_dir` to choose another location or to drive from outside the repository. Adjust `throttle`, `max_steps`, or
`telemetry_timeout` as keyword arguments. To enable camera lane assistance
on Generated Road, pass `lane_assist=True`. Interrupt the kernel to stop.

## Acceptance and Experiment Records

Verify on at least one Mac and one Windows machine: launch the simulator, connect
from Python, receive images, control steering and throttle, and close cleanly.
The Gymnasium API returns `(obs, info)` from `reset()` and
`(obs, reward, terminated, truncated, info)` from `step()`.

Launch the simulator executable and leave it running, then run this Python
snippet with the repository interpreter. The default simulator port is 9091.
The interface loads the track automatically. Use a process timeout when running
unattended, since the upstream connection and telemetry waits can block.

```python
import gymnasium as gym
import gym_donkeycar  # Registers the environments.
import numpy as np

env = gym.make("donkey-generated-roads-v0",
               conf={"host": "127.0.0.1", "port": 9091})
try:
    obs, info = env.reset(seed=0)
    assert obs.shape == (120, 160, 3) and obs.dtype == np.uint8
    for steering in (0.0, 0.15, -0.15):
        for _ in range(10):
            action = np.array([steering, 0.1], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                obs, info = env.reset()
    print("Connection, camera frames, and control steps OK")
finally:
    env.close()
```

Close the simulator window after the check. For unattended Windows checks,
launch with `-batchmode -nographics` and stop the process explicitly afterward;
this checks the protocol and frame shape, not visual rendering quality.

For each experiment, record the code commit, dependency versions, simulator version,
track, random seed, evaluation metrics, and inference time.
