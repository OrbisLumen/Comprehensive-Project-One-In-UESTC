import json

import numpy as np
from PIL import Image

from donkeycar_app import process_seeded
from donkeycar_app.dataset import preprocess_camera, read_split


def test_read_legacy_split(tmp_path):
    sample = {"route": "a", "image": "frame.png", "steering": 0.25}
    index = {"samples": [sample], "path_base": "repository_root", "sample_count": 1}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    assert read_split(path) == [dict(sample, raw=0.25, target=0.25)]


def test_process_native_recordings_and_read_training_labels(tmp_path, monkeypatch):
    config = {"raw_dir": "raw", "index_dir": "indices",
              "train": [1], "validation": [2], "test": [3]}
    manifest = tmp_path / "examples/donkeycar/road-seed-splits.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(config), encoding="utf-8")

    for seed in (1, 2, 3):
        folder = tmp_path / "raw" / f"seed-{seed}-drive"
        folder.mkdir(parents=True)
        metadata = {"seed": seed, "frames": 3, "layout_hash": f"layout-{seed}"}
        (folder / "collection.json").write_text(json.dumps(metadata), encoding="utf-8")
        for frame, angle in enumerate((0.25, -1.0, 0.5)):
            image = f"{frame}.jpg"
            Image.new("RGB", (6, 4), (seed * 60, frame * 60, 0)).save(folder / image)
            record = {"cam/image_array": image, "user/angle": angle, "user/throttle": 0.2}
            (folder / f"record_{frame}.json").write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr(process_seeded, "ROOT", tmp_path)
    process_seeded.main()

    for split in ("train", "validation", "test"):
        rows = read_split(tmp_path / "indices" / f"{split}.json")
        assert [row["raw"] for row in rows] == [0.25, -1.0, 0.5]
        assert [row["spike"] for row in rows] == [False, True, False]
        assert [row["target"] for row in rows] == [0.25, 0.375 if split == "train" else -1.0, 0.5]
        assert [row["weight"] for row in rows] == [1.0, 0.1 if split == "train" else 1.0, 1.0]
        with Image.open(tmp_path / rows[1]["image"]) as image:
            pixels = preprocess_camera(image)
        assert pixels.shape == (3, 4, 6)
        assert pixels.dtype == np.float32
