import csv
import io
import socket
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from donkeycar_app import drive as drive_app
from donkeycar_app.drive import drive
import json
from pathlib import Path
from mydezero import Variable, cuda


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_steer_transfers_rgb_frame_to_head_device(device):
    if device == "cuda" and not (cuda.gpu_enable and cuda.cupy.is_available()):
        pytest.skip("CUDA unavailable")
    xp = cuda.cupy if device == "cuda" else np

    class Probe:
        fc6 = SimpleNamespace(W=Variable(xp.zeros(1, dtype=xp.float32)))

        def __call__(self, pixels):
            assert cuda.get_array_module(pixels) is xp
            assert pixels.shape == (1, 3, 120, 160)
            assert pixels.dtype == xp.float32 and pixels.flags.c_contiguous
            return Variable(xp.asarray(0.25))

    notebook = Path(__file__).resolve().parents[2] / "examples/donkeycar/resnet50_single_frame_spatial_9728.ipynb"
    cell = next(c for c in json.loads(notebook.read_text("utf-8"))["cells"] if c["id"] == "steering-model")
    namespace = {}
    exec("".join(cell["source"]), namespace)
    SpatialSteeringResNet = namespace["SpatialSteeringResNet"]
    assert SpatialSteeringResNet.steer(Probe(), np.zeros((120, 160, 3), dtype=np.uint8)) == 0.25


def test_drive_uses_new_observation_until_episode_ends():
    class Env:
        def reset(self):
            return 0.1, {}

        def step(self, action):
            actions.append(action)
            done = len(actions) == 2
            return -0.2, 0, done, False, dict(cte=0, speed=1, hit="none", lap_count=0)

    actions = []
    log = io.StringIO()
    drive(Env(), float, 0.15, 10, log)
    np.testing.assert_allclose(actions, [[0.1, 0.15], [-0.2, 0.15]])
    log.seek(0)
    assert [row["terminated"] for row in csv.DictReader(log)] == ["False", "True"]


@pytest.mark.parametrize("offset_px", [-18, 18, None])
def test_lane_assist_corrects_toward_markings_and_falls_back(offset_px):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    if offset_px is not None:
        for y in (66, 72, 78):
            half_width = round(2.1 * (y - 42) / 2)
            center = 80 + offset_px
            frame[y, center - half_width] = (220, 170, 20)
            frame[y, center + half_width] = (230, 230, 230)
    prediction = -0.1 if offset_px == 18 else 0.1
    sent = []

    def step(action):
        sent.append(float(action[0]))
        return frame, 0, True, False, dict(cte=0, speed=1, hit="none", lap_count=0)

    env = SimpleNamespace(reset=lambda: (frame, {}), step=step)
    log = io.StringIO()
    drive(env, lambda image: prediction, 0.15, 1, log, lane_assist=True)
    log.seek(0)
    row = next(csv.DictReader(log))
    assert float(row["model_steering"]) == prediction
    assert float(row["steering"]) == pytest.approx(sent[0])
    if offset_px is None:
        assert row["lane_offset_px"] == "" and sent[0] == pytest.approx(prediction)
    else:
        assert float(row["lane_offset_px"]) == pytest.approx(offset_px + 0.5)
        assert np.sign(sent[0]) == np.sign(offset_px)
        assert abs(sent[0] - prediction) == pytest.approx(0.4)


@pytest.mark.parametrize("failure, message, brake_expected", [
    ("abort", "DonkeySim disconnected", False),
    ("silent", "No new telemetry from DonkeySim", True),
])
@pytest.mark.parametrize("custom_log_dir", [False, True])
def test_cli_exits_on_telemetry_failure_and_preserves_csv(tmp_path, monkeypatch,
                                                        capsys, failure, message, brake_expected, custom_log_dir):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src/donkeycar_app").mkdir(parents=True)
    notebook_dir = tmp_path / "examples/donkeycar"
    notebook_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path if failure == "abort" else notebook_dir)
    log_dir = tmp_path / "runs/donkeycar/closed-loop"
    options = {}
    if custom_log_dir:
        log_dir = tmp_path / "custom"
        options["log_dir"] = log_dir
    cleanup = []
    client = SimpleNamespace(
        aborted=False, do_process_msgs=True, th=SimpleNamespace(is_alive=lambda: True),
        s=SimpleNamespace(shutdown=lambda how: cleanup.append(("shutdown", how))),
    )
    handler = SimpleNamespace(last_received=0, time_received=0, fns={},
                              blocking_send=lambda msg: cleanup.append(("brake", msg["brake"])))
    info = dict(cte=0.1, speed=1, hit="none", lap_count=0)

    def observe():
        assert handler.last_received != handler.time_received
        handler.last_received = handler.time_received
        return np.full((2, 3, 3), handler.time_received, dtype=np.uint8), 0.0, False, info

    handler.observe = observe
    viewer = SimpleNamespace(client=client, handler=handler)

    class Env:
        unwrapped = SimpleNamespace(viewer=viewer)
        steps = 0

        def reset(self):
            handler.time_received += 1
            return handler.observe()[0], {}

        def step(self, action):
            self.steps += 1
            if self.steps == 1:
                handler.time_received += 1
            elif failure == "abort":
                handler.fns["aborted"]({})
            frame, reward, done, telemetry = handler.observe()
            return frame, reward, done, False, telemetry

        def close(self):
            assert not client.do_process_msgs
            cleanup.append(("close", None))

    env = Env()
    monkeypatch.setattr(drive_app.gym, "make", lambda *args, **kwargs: env)
    with pytest.raises(SystemExit) as error:
        drive_app.run_drive(lambda frame: 0.25, log_prefix="test_model-2x", telemetry_timeout=0.025, **options)
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert message in stderr
    [run_dir] = log_dir.glob("test_model-2x-*")
    log = run_dir / f"{run_dir.name}.csv"
    assert str(log) in stderr
    with log.open(newline="", encoding="utf-8") as source:
        [row] = csv.DictReader(source)
    assert row["step"] == "1" and float(row["steering"]) == 0.25
    [image] = run_dir.glob("*.png")
    assert image.name == row["image"]
    with Image.open(image) as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.ones((2, 3, 3), dtype=np.uint8))
    expected = [("brake", "1")] if brake_expected else []
    assert cleanup == expected + [("shutdown", socket.SHUT_RDWR), ("close", None)]


def test_select_checkpoint_filters_by_notebook_and_accepts_external_path(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src/donkeycar_app").mkdir(parents=True)
    for name in ("spatial-run/raw.npz", "spatial-run/raw.partial.npz", "global-run/raw.npz"):
        path = tmp_path / "models/donkeycar" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.chdir(tmp_path / "src")
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert drive_app.select_checkpoint("spatial") == tmp_path / "models/donkeycar/spatial-run/raw.npz"
    assert capsys.readouterr().out.splitlines() == [f"1. {Path('spatial-run/raw.npz')}"]
    external = tmp_path / "custom.npz"
    external.touch()
    monkeypatch.setattr("builtins.input", lambda _: str(external))
    assert drive_app.select_checkpoint("spatial") == external


@pytest.mark.parametrize("kwargs,message", [
    (dict(throttle=0.15, target_speed=2), "mutually exclusive"),
    (dict(target_speed=0), "positive target_speed"),
    (dict(target_speed=-1), "positive target_speed"),
    (dict(target_speed=float("nan")), "positive target_speed"),
    (dict(target_speed=float("inf")), "positive target_speed"),
])
def test_invalid_speed_options_fail_before_connection(kwargs, message):
    with pytest.raises(ValueError, match=message):
        drive_app.run_drive(float, **kwargs)


def test_speed_feedback_uses_latest_telemetry_and_logs_applied_throttle():
    speeds = iter([1.0, 2.5, 0.5])
    actions = []

    def step(action):
        actions.append(action)
        return 0.2, 0, False, False, dict(cte=0, speed=next(speeds), hit="none", lap_count=0)

    env = SimpleNamespace(reset=lambda: (0.2, {"speed": 0.0}), step=step)
    log = io.StringIO()
    drive(env, float, None, 3, log, target_speed=2.0)
    log.seek(0)
    rows = list(csv.DictReader(log))
    throttle = [float(row["throttle"]) for row in rows]
    np.testing.assert_allclose(np.array(actions)[:, 1], throttle)
    assert 0 == throttle[2] < throttle[1] < throttle[0] <= 1
    assert [float(row["speed"]) for row in rows] == [1, 2.5, 0.5]
    assert all(float(row["target_speed"]) == 2 for row in rows)


def test_speed_control_holds_cruise_throttle_and_recovers_from_saturation():
    control = drive_app._SpeedController(10)
    assert control.update(0, 60) == 1
    assert control.update(12, 1) == 0
    cruise = control.update(9, 1)
    assert 0 < control.update(10, 1) < cruise < 1
    with pytest.raises(ValueError, match="Invalid simulator speed"):
        control.update(float("nan"), 1)
