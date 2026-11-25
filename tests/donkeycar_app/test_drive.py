import csv
import io
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from donkeycar_app import drive as drive_app
from donkeycar_app.drive import drive
from donkeycar_app.spatial_resnet import SpatialSteeringResNet
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
    drive(Env(), SimpleNamespace(steer=float), 0.15, 10, log)
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
    drive(env, SimpleNamespace(steer=lambda image: prediction), 0.15, 1, log, lane_assist=True)
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
def test_cli_exits_on_telemetry_failure_and_preserves_csv(tmp_path, monkeypatch,
                                                        capsys, failure, message, brake_expected):
    checkpoint = tmp_path / "steering.npz"
    checkpoint.touch()
    log_dir = tmp_path / "runs"
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
        return None, 0.0, False, info

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
    model = SimpleNamespace(load_weights=lambda path: None, steer=lambda frame: 0.25)
    monkeypatch.setattr(drive_app, "SpatialSteeringResNet", lambda: model)
    monkeypatch.setattr(drive_app.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(drive_app.sys, "argv", [
        "drive", "--cpu", "--model-only", "--checkpoint", str(checkpoint),
        "--log-dir", str(log_dir), "--telemetry-timeout", "0.025",
    ])

    with pytest.raises(SystemExit) as error:
        drive_app.main()
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert message in stderr
    [log] = log_dir.glob("*.csv")
    assert str(log) in stderr
    with log.open(newline="", encoding="utf-8") as source:
        [row] = csv.DictReader(source)
    assert row["step"] == "1" and float(row["steering"]) == 0.25
    expected = [("brake", "1")] if brake_expected else []
    assert cleanup == expected + [("shutdown", socket.SHUT_RDWR), ("close", None)]
