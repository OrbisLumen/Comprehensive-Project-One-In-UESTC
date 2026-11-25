import csv
import io
from types import SimpleNamespace

import numpy as np
import pytest

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

