import csv

import numpy as np
import pytest
from PIL import Image

from donkeycar_app.record import Recording


def test_recording_pairs_pre_step_frame_with_action_and_skips_idle(tmp_path):
    frame = np.full((2, 3, 3), 42, dtype=np.uint8)

    class FakeEnv:
        def step(self, action):
            frame[:] = 84
            return frame, 0.0, True, False, {}

    (tmp_path / "images").mkdir()
    with (tmp_path / "samples.csv").open("w+", encoding="utf-8", newline="") as samples:
        recording = Recording(tmp_path, samples)
        action = np.array([-0.5, 0.125], dtype=np.float32)
        next_frame, _, terminated, _, _ = recording.step(FakeEnv(), frame, action)
        recording.step(FakeEnv(), next_frame, np.zeros(2, dtype=np.float32))

        assert terminated
        assert recording.count == 1
        samples.seek(0)
        rows = list(csv.DictReader(samples))

    assert rows == [{"image": "images/000000.png", "steering": "-0.5", "throttle": "0.125"}]
    with Image.open(tmp_path / rows[0]["image"]) as image:
        np.testing.assert_array_equal(np.asarray(image), np.full_like(frame, 42))


def test_recording_does_not_write_sample_when_step_fails(tmp_path):
    class DisconnectedEnv:
        def step(self, action):
            raise ConnectionError("simulator disconnected")

    (tmp_path / "images").mkdir()
    with (tmp_path / "samples.csv").open("w+", encoding="utf-8", newline="") as samples:
        recording = Recording(tmp_path, samples)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        with pytest.raises(ConnectionError):
            recording.step(DisconnectedEnv(), frame, np.array([0, 0.125], dtype=np.float32))

        samples.seek(0)
        assert list(csv.DictReader(samples)) == []
        assert list((tmp_path / "images").iterdir()) == []
