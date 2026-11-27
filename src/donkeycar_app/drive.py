"""Drive DonkeySim with an RGB-frame steering predictor."""

import re
import csv
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import socket
import sys
from time import perf_counter, sleep
from uuid import uuid4

import gymnasium as gym
import gym_donkeycar  # Registers the simulator environments.
import numpy as np
from PIL import Image

from donkeycar_app.record import DEFAULT_TRACK


DEFAULT_TELEMETRY_TIMEOUT = 5.0


def _project_root():
    cwd = Path.cwd()
    return next((p for p in (cwd, *cwd.parents)
                 if (p / "pyproject.toml").is_file() and (p / "src/donkeycar_app").is_dir()), None)


def select_checkpoint(notebook_name):
    """Choose a saved notebook checkpoint by number or enter a file path."""
    root = _project_root()
    candidates = []
    if root is not None:
        model_dir = root / "models/donkeycar"
        candidates = sorted(path for path in model_dir.glob(f"{notebook_name}-*/*.npz")
                            if path.is_file() and not path.name.endswith(".partial.npz"))
        for number, path in enumerate(candidates, start=1):
            print(f"{number}. {path.relative_to(model_dir)}")
    choice = input("Checkpoint number or .npz path: ").strip()
    if choice.isdecimal():
        number = int(choice)
        if 1 <= number <= len(candidates):
            return candidates[number - 1]
        raise ValueError("Choose a listed checkpoint number or enter a .npz path.")
    path = Path(choice).expanduser()
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an existing .npz checkpoint: {choice!r}")
    return path.resolve()


def telemetry_connected(client):
    return not client.aborted and client.do_process_msgs and client.th.is_alive()


def configure_telemetry(viewer, timeout):
    """Bound frame waits for this instance of the pinned simulator interface."""
    handler, client = viewer.handler, viewer.client
    observe = handler.observe

    def on_abort(_message):
        # The receiver cannot join itself. The driving thread owns socket cleanup.
        client.aborted = True
        client.do_process_msgs = False

    def observe_with_timeout():
        deadline = perf_counter() + timeout
        while True:
            if not telemetry_connected(client):
                raise ConnectionError("DonkeySim disconnected while waiting for telemetry.")
            if handler.last_received != handler.time_received:
                # The driving thread is the only consumer of this frame marker.
                return observe()
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"No new telemetry from DonkeySim for {timeout:g} seconds.")
            sleep(min(0.001, remaining))

    handler.fns["aborted"] = on_abort
    handler.observe = observe_with_timeout


def close_simulator(env, viewer):
    """Attempt a final brake and wake the receiver before closing the environment."""
    client = viewer.client
    connected = telemetry_connected(client)
    client.do_process_msgs = False
    try:
        if connected:
            viewer.handler.blocking_send({
                "msg_type": "control", "steering": "0", "throttle": "0", "brake": "1"
            })
    except OSError:
        pass  # The peer may have closed after the connection check.
    finally:
        try:
            client.s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # A disconnected socket may already be shut down.
        finally:
            env.close()


def lane_offset(frame):
    """Estimate the right lane's look-ahead center offset in a 160 x 120 Generated Road image."""
    offsets = []
    for y in (66, 72, 78):
        row = frame[y].astype(np.int16)
        red, green, blue = row.T
        yellow = np.flatnonzero((red > 135) & (red - blue > 65) & (green - blue > 25))
        if not yellow.size or yellow[-1] - yellow[0] > 18:
            continue
        left = float(np.median(yellow))
        # Perspective width of the right lane at this row in the default camera.
        width = 2.1 * (y - 42)
        white = np.flatnonzero((row.min(axis=1) > 145) & (np.ptp(row, axis=1) < 40))
        right = white[(white > left + width * 0.65) & (white < left + width * 1.4)]
        center = (left + np.median(right)) / 2 if right.size else left + width / 2
        offsets.append(center - (frame.shape[1] - 1) / 2)
    return float(np.median(offsets)) if len(offsets) >= 2 else None


class _SpeedController:
    """PI throttle control using elapsed seconds between observations."""

    def __init__(self, target_speed):
        self.target_speed = target_speed
        self.cruise_throttle = 0.0

    def update(self, speed, dt):
        if not isfinite(speed) or speed < 0:
            raise ValueError(f"Invalid simulator speed: {speed}")
        error = self.target_speed - speed
        proportional = 0.15 * error
        # Freeze integration when the current error would drive a saturated output further.
        output = proportional + self.cruise_throttle
        if not ((output >= 1 and error > 0) or (output <= 0 and error < 0)):
            self.cruise_throttle = float(np.clip(self.cruise_throttle + 0.05 * error * dt, 0, 1))
        return float(np.clip(proportional + self.cruise_throttle, 0, 1))


def drive(env, steer_frame, throttle, max_steps, log, lane_assist=False, image_dir=None, *, target_speed=None):
    """Run one episode and log the action and resulting telemetry for each step."""
    writer = csv.writer(log)
    writer.writerow(("step", "elapsed_s", "steering", "throttle", "inference_ms",
                     "cte", "speed", "hit", "lap_count", "terminated", "truncated",
                     "model_steering", "lane_offset_px", "target_speed", "image"))
    controller = _SpeedController(target_speed) if target_speed is not None else None
    frame, info = env.reset()
    start = perf_counter()
    previous_update = start
    for step in range(1, max_steps + 1):
        before = perf_counter()
        prediction = steer_frame(frame)
        latency = (perf_counter() - before) * 1000
        if not -1 <= prediction <= 1:
            raise ValueError(f"Invalid predicted steering: {prediction}")
        offset = lane_offset(frame) if lane_assist else None
        correction = 0.0 if offset is None else np.clip(0.025 * offset, -0.4, 0.4)
        steer = float(np.clip(prediction + correction, -1, 1))
        if controller is not None:
            now = perf_counter()
            throttle = controller.update(info["speed"], now - previous_update)
            previous_update = now
        action = np.array([steer, throttle], dtype=np.float32)
        image = Image.fromarray(frame).copy() if image_dir is not None else None
        frame, _, terminated, truncated, info = env.step(action)
        image_name = ""
        if image is not None:
            image_name = f"frame-{step:06d}.png"
            image.save(image_dir / image_name)
        writer.writerow((step, round(perf_counter() - start, 4), steer, throttle, round(latency, 3),
                         info["cte"], info["speed"], info["hit"], info["lap_count"], terminated, truncated,
                         prediction, offset, target_speed, image_name))
        log.flush()
        if step == 1 or step % 50 == 0 or terminated or truncated:
            print(f"Step {step}  steer {steer:+.3f}  CTE {info['cte']:+.2f}  "
                  f"speed {info['speed']:.2f}  inference {latency:.1f} ms", flush=True)
        if terminated or truncated:
            return f"Episode ended at step {step} (hit={info['hit']}, CTE={info['cte']:.2f})."
    return f"Reached the limit of {max_steps} steps."


def run_drive(steer_frame, *, track=DEFAULT_TRACK, host="127.0.0.1", port=9091,
              throttle=None, target_speed=None, max_steps=2000, telemetry_timeout=DEFAULT_TELEMETRY_TIMEOUT,
              log_dir=None, lane_assist=False, log_prefix=None):
    """Drive one episode using a callable from RGB camera frame to steering."""
    if target_speed is not None:
        if throttle is not None:
            raise ValueError("throttle and target_speed are mutually exclusive.")
        if not isfinite(target_speed) or target_speed <= 0:
            raise ValueError("Use a finite positive target_speed.")
    else:
        throttle = 0.15 if throttle is None else throttle
        if not 0 < throttle <= 1:
            raise ValueError("Use throttle in (0, 1].")
    if max_steps < 1:
        raise ValueError("Use a positive step limit.")
    if not isfinite(telemetry_timeout) or telemetry_timeout <= 0:
        raise ValueError("Use a finite positive telemetry timeout.")
    if log_prefix is None:
        log_prefix = getattr(getattr(steer_frame, "__self__", None), "drive_log_prefix", None)
    if not log_prefix or not re.fullmatch(r"[A-Za-z0-9_-]+", log_prefix):
        raise ValueError("Supply a filename-safe log_prefix identifying the notebook and multiplier.")
    if log_dir is None:
        root = _project_root()
        if root is None:
            raise ValueError("Pass log_dir when driving outside the project repository.")
        log_dir = root / "runs/donkeycar/closed-loop"
    steer_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    log_dir = Path(log_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    run_dir = log_dir / f"{log_prefix}-{run_id}"
    run_dir.mkdir(parents=True)
    log_path = run_dir / f"{run_dir.name}.csv"
    print(f"Connecting to {host}:{port} for {track}.", flush=True)
    env = gym.make(track, conf={"host": host, "port": port})
    viewer = env.unwrapped.viewer
    try:
        configure_telemetry(viewer, telemetry_timeout)
        with log_path.open("x", encoding="utf-8", newline="") as log:
            assisted = track == DEFAULT_TRACK and lane_assist
            mode = "Model + camera lane assistance" if assisted else "Model only"
            control = f"throttle {throttle}" if target_speed is None else f"target speed {target_speed}"
            print(f"{mode}, {control}. Ctrl+C to stop. Log: {log_path}", flush=True)
            print(drive(env, steer_frame, throttle, max_steps, log, lane_assist=assisted,
                        image_dir=run_dir, target_speed=target_speed))
    except KeyboardInterrupt:
        print("Drive interrupted.")
    except (ConnectionError, TimeoutError) as error:
        print(f"{error} Completed steps are saved in {log_path}.", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        close_simulator(env, viewer)


if __name__ == "__main__":
    sys.exit("Load your checkpoint in its single-frame ResNet50 notebook and call run_drive(model.steer).")
