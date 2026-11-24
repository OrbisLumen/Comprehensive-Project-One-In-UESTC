"""Run the course steering model against a running Donkey simulator."""

import argparse
import csv
from math import hypot
from pathlib import Path

import gymnasium as gym
import gym_donkeycar  # Registers the simulator environments.
import numpy as np
import torch

from donkeycar_app.autodrivenet import AutoDriveNet
from donkeycar_app.record import DEFAULT_TRACK


def drive_episode(env, model, throttle, steps, output):
    """Log applied steering, lateral deviation and horizontal path length."""
    frame, info = env.reset()
    if frame.shape != (120, 160, 3) or np.ptp(frame) == 0:
        raise ValueError("Simulator did not return a rendered camera frame")
    x, _, z = info["pos"]
    distance = 0.0
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "steering", "throttle", "x", "z", "path_length", "cte",
                         "speed", "hit", "lap_count", "terminated", "truncated"))
        for step in range(1, steps + 1):
            steering = model.steer(frame)
            action = np.array([steering, throttle], dtype=np.float32)
            frame, _, terminated, truncated, info = env.step(action)
            next_x, _, next_z = info["pos"]
            distance += hypot(next_x - x, next_z - z)
            x, z = next_x, next_z
            writer.writerow((step, steering, throttle, x, z, distance, info["cte"],
                             info["speed"], info["hit"], info["lap_count"], terminated, truncated))
            if terminated or truncated or info["lap_count"] > 0:
                break
    return dict(steps=step, path_length=distance, laps=info["lap_count"], cte=info["cte"],
                hit=info["hit"], terminated=bool(terminated), truncated=bool(truncated))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--throttle", type=float, default=0.20)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.throttle <= 1:
        parser.error("Use positive steps and throttle in (0, 1]")
    model = AutoDriveNet().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    torch.set_num_threads(1)
    env = gym.make(DEFAULT_TRACK, conf={"host": "127.0.0.1", "port": args.port})
    try:
        print(drive_episode(env, model, args.throttle, args.steps, args.out))
    finally:
        try:
            env.unwrapped.viewer.handler.blocking_send(
                {"msg_type": "control", "steering": "0", "throttle": "0", "brake": "1"}
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
