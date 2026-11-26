import json

import numpy as np
import pytest
from PIL import Image

from donkeycar_app.dataset import SteeringDataset


@pytest.mark.parametrize("legacy", [False, True])
def test_split_returns_training_label_and_native_image_size(tmp_path, legacy):
    Image.new("RGB", (6, 4)).save(tmp_path / "frame.png")
    sample = {"route": "a", "image": "frame.png"}
    if legacy:
        sample["steering"] = 0.25
        index = {"samples": [sample], "path_base": "repository_root", "sample_count": 1}
    else:
        index = [dict(sample, raw=-1.0, target=0.25)]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    pixels, target = SteeringDataset(tmp_path, path, ["a"])[0]
    assert pixels.shape == (3, 4, 6)
    np.testing.assert_array_equal(target, np.array([0.25], dtype=np.float32))
