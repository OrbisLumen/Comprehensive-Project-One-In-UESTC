import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from donkeycar_app.autodrivenet import AutoDriveNet, hsv_frame
from donkeycar_app.train_autodrivenet import batch_images


def test_rgb_primaries_have_opencv_hues():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[0, :3] = np.eye(3, dtype=np.uint8) * 255
    assert hsv_frame(rgb)[0, :3].tolist() == [[0, 255, 255], [60, 255, 255], [120, 255, 255]]


def test_live_steering_matches_training_input_and_clips():
    rgb = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    model = AutoDriveNet().eval()
    images = batch_images(torch.from_numpy(hsv_frame(rgb)[None]), "cpu")
    with torch.inference_mode():
        expected = model(images).clamp(-1, 1).item()
        assert model.steer(rgb) == pytest.approx(expected, abs=1e-6)
        model.regressor[-1].weight.zero_()
        model.regressor[-1].bias.fill_(2)
    assert model.steer(rgb) == 1.0
