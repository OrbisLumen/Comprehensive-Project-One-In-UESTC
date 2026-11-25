"""Select a seeded road and archive one built-in Auto Drive w Rec session."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Queue
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from uuid import uuid4

import gymnasium as gym
import gym_donkeycar  # Registers the DonkeyCar environments.


class LayoutMismatch(RuntimeError):
    pass


NODE_INDICES = (1, 5, 10, 20, 30)


def show_seed(port, output, track):
    """Inspect an already loaded scene without issuing scene or control commands."""
    points = []
    with socket.create_connection(("127.0.0.1", port), timeout=8) as sock, sock.makefile("rb") as stream:
        for index in NODE_INDICES:
            deadline = time.monotonic() + 8
            sent = False
            while time.monotonic() < deadline:
                line = stream.readline()
                if not line:
                    raise ConnectionError("Simulator disconnected")
                msg = json.loads(line)
                kind = msg.get("msg_type")
                if kind == "scene_selection_ready":
                    raise ValueError("Open a road in the simulator before querying it.")
                if not sent and kind in ("car_loaded", "need_car_config", "telemetry"):
                    sock.sendall(json.dumps(dict(msg_type="node_position", index=str(index))).encode())
                    sent = True
                if kind == "node_position" and int(msg["index"]) == index:
                    points.append([index, *[round(float(msg[k]), 3) for k in ("pos_x", "pos_y", "pos_z")]])
                    break
            else:
                raise TimeoutError(f"No reply for road node {index}")
    digest = hashlib.sha256(json.dumps(points).encode()).hexdigest()
    matches = set()
    for path in output.glob("*/collection.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("track") == track and record.get("sampled_nodes") == points:
            matches.add(record["seed"])
    print("Sampled layout:", digest)
    print("Matching archived seeds:", sorted(matches) if matches else "unknown")


def sample_layout(handler):
    replies = Queue()
    handler.fns["node_position"] = replies.put
    points = []
    for index in NODE_INDICES:
        handler.blocking_send({"msg_type": "node_position", "index": str(index)})
        reply = replies.get(timeout=5)
        if int(reply["index"]) != index:
            raise RuntimeError("Unexpected road node reply")
        points.append([index, *[round(float(reply[k]), 3) for k in ("pos_x", "pos_y", "pos_z")]])
    return points


def select_layout(handler, seed, style, turn):
    def rebuild(value):
        handler.blocking_send(dict(msg_type="regen_road", road_style=str(style),
                                   rand_seed=str(value), turn_increment=str(turn)))
        time.sleep(1)
        points = sample_layout(handler)
        digest = hashlib.sha256(json.dumps(points).encode()).hexdigest()[:12]
        print(f"Road seed {value}, node signature {digest}", flush=True)
        return points

    first = rebuild(seed)
    other = rebuild(seed ^ 1)
    final = rebuild(seed)
    if first != final:
        raise LayoutMismatch(f"Seed {seed} produced different nodes on repeat.")
    if first == other:
        raise LayoutMismatch(f"Seeds {seed} and {seed ^ 1} produced identical sampled nodes.")
    return final


def archive(log, output, metadata):
    records = sorted(log.glob("record_*.json"), key=lambda p: int(p.stem[7:]))
    if not records:
        raise ValueError(f"No native records in {log}. Check the simulator Log dir.")
    output.mkdir(parents=True, exist_ok=True)
    name = f"seed-{metadata['seed']}-{uuid4().hex[:8]}"
    with TemporaryDirectory(dir=output) as tmp:
        staging = Path(tmp) / name
        staging.mkdir()
        shutil.copy2(log / "meta.json", staging)
        for record in records:
            row = json.loads(record.read_text(encoding="utf-8-sig"))
            image = row["cam/image_array"]
            if Path(image).name != image or not image.lower().endswith(".jpg"):
                raise ValueError(f"Unexpected native image name in {record}")
            shutil.copy2(log / image, staging)
            shutil.copy2(record, staging)
        metadata["frames"] = len(records)
        (staging / "collection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(output / name)
    return output / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="donkey-generated-roads-v0", help="Registered Gym track ID")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--show-seed", action="store_true", help="Look up the current layout in archived recordings")
    parser.add_argument("--road-style", type=int, default=0)
    parser.add_argument("--turn-increment", type=float, default=0)
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--sim-exe", type=Path, default=Path(__file__).resolve().parents[2] /
                        "tools/donkey-sim/DonkeySimWin/donkey_sim.exe",
                        help="Simulator executable for a fresh session if layout verification fails")
    parser.add_argument("--log-dir", type=Path, help="Native simulator Log dir for this recording")
    parser.add_argument("--output", type=Path, default=Path("data/raw/donkeycar-seeded"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--preview", action="store_true",
                        help="Keep the verified road open without recording or driving")
    args = parser.parse_args()
    if args.show_seed:
        if args.preview or args.check_only or args.log_dir:
            parser.error("--show-seed is a standalone query")
        try:
            show_seed(args.port, args.output, args.track)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Seed query failed: {error}\n")
        return
    if args.preview and not args.check_only:
        parser.error("--preview requires --check-only")
    if not 0 <= args.seed <= 2147483647:
        parser.error("Use a nonnegative 32-bit seed")
    if not args.check_only and (args.log_dir is None or any(args.log_dir.glob("record_*.json"))):
        parser.error("Select a fresh simulator Log dir and pass the same path with --log-dir")
    proc = None
    conf = {"host": "127.0.0.1", "port": args.port}
    if args.check_only:
        env = gym.make(args.track, conf=conf)
    else:
        from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimHandler

        input(f"In DonkeySim, manually enter the scene for {args.track}. "
              "Confirm Auto Drive w Rec is visible, then press Enter here: ")
        def require_manual_scene(handler, message):
            # An automatic load_scene permanently hides Panel Menu for the scene.
            print("Simulator is at the main menu. Select the requested track in its window.", flush=True)

        with patch.object(DonkeyUnitySimHandler, "on_recv_scene_names", require_manual_scene):
            env = gym.make(args.track, conf=conf)
    try:
        handler = env.unwrapped.viewer.handler
        try:
            points = select_layout(handler, args.seed, args.road_style, args.turn_increment)
        except LayoutMismatch as error:
            if not args.check_only:
                raise RuntimeError(f"{error} Manual recording layout is unverified. "
                                   "Recording cannot be assigned this road seed.") from error
            if not args.sim_exe.is_file():
                raise RuntimeError(f"{error} Supply --sim-exe to retry in a fresh simulator.") from error
            env.close()
            env = None
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            print(f"{error} Starting a fresh simulator on port {port}. Use its window for recording.", flush=True)
            proc = subprocess.Popen([str(args.sim_exe.resolve()), "--port", str(port), "--host", "127.0.0.1"])
            # A TCP readiness probe itself creates a simulator client and can load the scene twice.
            time.sleep(3)
            if proc.poll() is not None:
                raise RuntimeError("Fresh simulator failed to start")
            env = gym.make(args.track, conf={"host": "127.0.0.1", "port": port})
            handler = env.unwrapped.viewer.handler
            points = select_layout(handler, args.seed, args.road_style, args.turn_increment)
        fingerprint = hashlib.sha256(json.dumps(points).encode()).hexdigest()
        print(f"Seed {args.seed}, sampled layout {fingerprint}", flush=True)
        if args.check_only:
            if args.preview:
                try:
                    input("Verified road is open. Press Enter to exit: ")
                except KeyboardInterrupt:
                    pass
            return
        input("Click Auto Drive w Rec, then Stop after one drive. Press Enter here after recording stops: ")
        if sample_layout(handler) != points:
            raise RuntimeError("Road changed during the session. Native files remain in the simulator Log dir.")
        metadata = dict(track=args.track, seed=args.seed, road_style=args.road_style,
                        turn_increment=args.turn_increment, sampled_nodes=points, layout_hash=fingerprint,
                        recorded_utc=datetime.now(timezone.utc).isoformat())
        print(archive(args.log_dir, args.output, metadata))
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
