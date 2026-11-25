"""Drive the DonkeyCar simulator with a trained spatial ResNet50."""

import argparse
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

from donkeycar_app.record import DEFAULT_TRACK
from donkeycar_app.spatial_resnet import SpatialSteeringResNet
from mydezero import cuda


DEFAULT_CHECKPOINT = Path("models/donkeycar/20261123T145241Z-ee0bda/best.npz")
DEFAULT_TELEMETRY_TIMEOUT = 5.0


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


def drive(env, model, throttle, max_steps, log, lane_assist=False):
    """Run one episode and log the action and resulting telemetry for each step."""
    writer = csv.writer(log)
    writer.writerow(("step", "elapsed_s", "steering", "throttle", "inference_ms",
                     "cte", "speed", "hit", "lap_count", "terminated", "truncated",
                     "model_steering", "lane_offset_px"))
    frame, _ = env.reset()
    start = perf_counter()
    for step in range(1, max_steps + 1):
        before = perf_counter()
        prediction = model.steer(frame)
        latency = (perf_counter() - before) * 1000
        if not -1 <= prediction <= 1:
            raise ValueError(f"Invalid predicted steering: {prediction}")
        offset = lane_offset(frame) if lane_assist else None
        correction = 0.0 if offset is None else np.clip(0.025 * offset, -0.4, 0.4)
        steer = float(np.clip(prediction + correction, -1, 1))
        action = np.array([steer, throttle], dtype=np.float32)
        frame, _, terminated, truncated, info = env.step(action)
        writer.writerow((step, round(perf_counter() - start, 4), steer, throttle, round(latency, 3),
                         info["cte"], info["speed"], info["hit"], info["lap_count"], terminated, truncated,
                         prediction, offset))
        log.flush()
        if step == 1 or step % 50 == 0 or terminated or truncated:
            print(f"Step {step}  steer {steer:+.3f}  CTE {info['cte']:+.2f}  "
                  f"speed {info['speed']:.2f}  inference {latency:.1f} ms", flush=True)
        if terminated or truncated:
            return f"Episode ended at step {step} (hit={info['hit']}, CTE={info['cte']:.2f})."
    return f"Reached the limit of {max_steps} steps."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--throttle", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--telemetry-timeout", type=float, default=DEFAULT_TELEMETRY_TIMEOUT,
                        help="Maximum wait for a fresh frame in seconds (default: %(default)s).")
    parser.add_argument("--cpu", action="store_true", help="Use CPU even if CUDA is available.")
    parser.add_argument("--model-only", action="store_true", help="Disable Generated Road camera lane assistance.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/donkeycar/closed-loop"))
    args = parser.parse_args()
    if not 0 < args.throttle <= 1 or args.max_steps < 1:
        parser.error("Use throttle in (0, 1] and a positive step limit.")
    if not isfinite(args.telemetry_timeout) or args.telemetry_timeout <= 0:
        parser.error("--telemetry-timeout must be finite and positive.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}. Run from the repository root or set --checkpoint.")

    model = SpatialSteeringResNet()
    model.load_weights(args.checkpoint)
    use_gpu = not args.cpu and cuda.gpu_enable and cuda.cupy.is_available()
    if use_gpu:
        model.to_gpu()
    model.steer(np.zeros((120, 160, 3), dtype=np.uint8))  # Warm up before connecting.
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    log_path = args.log_dir / f"{run_id}.csv"
    print(f"Loaded {args.checkpoint} on {'CUDA' if use_gpu else 'CPU'}. "
          f"Connecting to {args.host}:{args.port} for {args.track}.", flush=True)
    env = gym.make(args.track, conf={"host": args.host, "port": args.port})
    viewer = env.unwrapped.viewer
    try:
        configure_telemetry(viewer, args.telemetry_timeout)
        with log_path.open("x", encoding="utf-8", newline="") as log:
            assisted = args.track == DEFAULT_TRACK and not args.model_only
            mode = "ResNet + camera lane assistance" if assisted else "ResNet only"
            print(f"{mode}, throttle {args.throttle}. Ctrl+C to stop. Log: {log_path}", flush=True)
            print(drive(env, model, args.throttle, args.max_steps, log, lane_assist=assisted))
    except KeyboardInterrupt:
        print("Drive interrupted.")
    except (ConnectionError, TimeoutError) as error:
        print(f"{error} Completed steps are saved in {log_path}.", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        close_simulator(env, viewer)


if __name__ == "__main__":
    main()
