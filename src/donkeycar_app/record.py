"""Record camera frames and keyboard driving actions from DonkeyCar."""

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image


DEFAULT_TRACK = "donkey-generated-roads-v0"
DEFAULT_OUTPUT = Path("data/raw/donkeycar")
UPDATE_MS = 33
STEERING_RATE = 2.0  # Steering units per second while a key is held.


class Recording:
    """Write numbered images and their actions into one recording directory."""

    def __init__(self, directory, samples_file):
        self.directory = directory
        self.samples_file = samples_file
        self.writer = csv.writer(samples_file)
        self.writer.writerow(("image", "steering", "throttle"))
        self.count = 0

    def step(self, env, observation, action):
        # Snapshot the displayed frame before step can advance the camera buffer.
        image = Image.fromarray(observation.copy()) if action[1] > 0 else None
        steering, throttle = map(float, action)
        transition = env.step(action)
        if image is not None:
            path = Path("images") / f"{self.count:06d}.png"
            image.save(self.directory / path)
            self.writer.writerow((path.as_posix(), steering, throttle))
            self.samples_file.flush()
            self.count += 1
        return transition


def drive(env, observation, recording, throttle):
    import tkinter as tk

    from PIL import ImageTk

    # A relocated Python prefix may still have the old Tcl/Tk path compiled in.
    library_dir = Path(sys.base_prefix) / "lib"
    tcl_dir = library_dir / f"tcl{tk.TclVersion}"
    tk_dir = library_dir / f"tk{tk.TkVersion}"
    if (tcl_dir / "init.tcl").is_file():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_dir))
    if (tk_dir / "tk.tcl").is_file():
        os.environ.setdefault("TK_LIBRARY", str(tk_dir))

    root = tk.Tk()
    try:
        root.title("DonkeyCar Recorder")
        tk.Label(root, text="Hold ↑ to drive, steer with ← / →, press Esc to finish.").pack()
        camera = tk.Label(root)
        camera.pack()
        status = tk.Label(root)
        status.pack()

        keys = set()
        root.bind("<KeyPress>", lambda event: keys.add(event.keysym))
        root.bind("<KeyRelease>", lambda event: keys.discard(event.keysym))
        root.bind("<FocusOut>", lambda event: keys.clear())
        root.bind("<Escape>", lambda event: root.quit())
        root.protocol("WM_DELETE_WINDOW", root.quit)

        steering = 0.0
        last_update = monotonic()
        error = None

        def show_frame():
            image = Image.fromarray(observation)
            image = image.resize((image.width * 3, image.height * 3))
            camera.image = ImageTk.PhotoImage(image, master=root)
            camera.configure(image=camera.image)
            status.configure(text=f"Frames saved: {recording.count}    Steering: {steering:+.2f}")

        def advance():
            nonlocal observation, steering, last_update, error
            now = monotonic()
            direction = int("Right" in keys) - int("Left" in keys)
            if direction:
                change = direction * STEERING_RATE * min(now - last_update, 0.1)
                steering = float(np.clip(steering + change, -1.0, 1.0))
            else:
                steering = 0.0
            last_update = now
            action = np.array([steering, throttle if "Up" in keys else 0.0], dtype=np.float32)

            try:
                observation, _, terminated, truncated, _ = recording.step(env, observation, action)
                if terminated or truncated:
                    print("Simulator episode ended.")
                    root.quit()
                    return
                show_frame()
            except Exception as exc:
                error = exc
                root.quit()
                return
            root.after(UPDATE_MS, advance)

        show_frame()
        root.after(UPDATE_MS, advance)
        root.mainloop()
        if error is not None:
            raise error
        env.step(np.zeros(2, dtype=np.float32))
    finally:
        root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=DEFAULT_TRACK, help="Gymnasium DonkeyCar environment ID")
    parser.add_argument("--host", default="127.0.0.1", help="simulator host")
    parser.add_argument("--port", type=int, default=9091, help="simulator port")
    parser.add_argument("--throttle", type=float, default=0.15, help="throttle while holding Up (default: 0.15)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="parent directory for recordings")
    args = parser.parse_args()
    if not 0.0 < args.throttle <= 1.0:
        parser.error("--throttle must be greater than 0 and at most 1")

    import gymnasium as gym
    import gym_donkeycar  # Registers the DonkeyCar environments.

    env = gym.make(args.track, conf={"host": args.host, "port": args.port})
    try:
        observation, _ = env.reset()
        prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=prefix, dir=args.output_dir))
        (directory / "images").mkdir()
        metadata = {"track": args.track, "throttle": args.throttle, "image_shape": list(observation.shape)}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Recording to {directory}")
        with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as samples_file:
            recording = Recording(directory, samples_file)
            drive(env, observation, recording, args.throttle)
        print(f"Saved {recording.count} frames in {directory}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
