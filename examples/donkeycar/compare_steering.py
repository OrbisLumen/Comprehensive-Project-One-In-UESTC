"""Alternate course CNN and existing MyDeZero ResNet in one loaded scene."""

import argparse
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

import donkeycar_app
from donkeycar_app.autodrivenet import AutoDriveNet
from donkeycar_app.drive_autodrivenet import drive_episode
from donkeycar_app.record import DEFAULT_TRACK


class CaptureStart(gym.Wrapper):
    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        self.start_frame = frame.copy()
        return frame, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", required=True, type=Path)
    parser.add_argument("--resnet", required=True, type=Path)
    parser.add_argument("--resnet-src", required=True, type=Path,
                        help="src directory containing the reference SpatialSteeringResNet and MyDeZero")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--throttle", type=float, default=0.20)
    args = parser.parse_args()
    if min(args.steps, args.pairs) < 1 or not 0 < args.throttle <= 1:
        parser.error("Use positive pairs, steps and throttle in (0, 1]")
    reference_src = args.resnet_src.resolve()
    sys.path.insert(0, str(reference_src))
    donkeycar_app.__path__.append(str(reference_src / "donkeycar_app"))
    from donkeycar_app.spatial_resnet import SpatialSteeringResNet
    from mydezero import cuda

    course = AutoDriveNet().eval()
    torch.set_num_threads(1)
    course.load_state_dict(torch.load(args.course, weights_only=True, map_location="cpu"))
    resnet = SpatialSteeringResNet()
    resnet.load_weights(args.resnet)
    if cuda.gpu_enable and cuda.cupy.is_available():
        resnet.to_gpu()
    models = {"course": course, "resnet": resnet}
    for model in models.values():
        model.steer(np.zeros((120, 160, 3), dtype=np.uint8))
    args.out.mkdir(parents=True, exist_ok=False)
    env = CaptureStart(gym.make(DEFAULT_TRACK, conf={"host": "127.0.0.1", "port": args.port}))
    try:
        for pair in range(1, args.pairs + 1):
            order = ("course", "resnet") if pair % 2 else ("resnet", "course")
            for name in order:
                tag = f"{pair}-{name}"
                result = drive_episode(env, models[name], args.throttle, args.steps, args.out / f"{tag}.csv")
                Image.fromarray(env.start_frame).save(args.out / f"{tag}-start.png")
                result.update(pair=pair, model=name, throttle=args.throttle, step_limit=args.steps)
                print(result, flush=True)
                with (args.out / "episodes.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(result) + "\n")
    finally:
        env.unwrapped.viewer.handler.blocking_send(
            {"msg_type": "control", "steering": "0", "throttle": "0", "brake": "1"}
        )
        env.close()


if __name__ == "__main__":
    main()
