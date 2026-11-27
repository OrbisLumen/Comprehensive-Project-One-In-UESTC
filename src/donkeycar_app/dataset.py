"""Read indexed simulator drives for steering regression."""

import json
from typing import NamedTuple

import numpy as np

from mydezero.models import VGG16


class SteeringRecipe(NamedTuple):
    label: str
    downweight_spikes: bool
    sharp_multiplier: int


STEERING_RECIPES = {
    "raw": SteeringRecipe("raw", False, 1),
    "raw_x2": SteeringRecipe("raw", False, 2),
    "raw_x4": SteeringRecipe("raw", False, 4),
    "downweighted_x1": SteeringRecipe("raw", True, 1),
    "downweighted_x2": SteeringRecipe("raw", True, 2),
    "downweighted_x4": SteeringRecipe("raw", True, 4),
    "smoothed_x1": SteeringRecipe("target", False, 1),
    "smoothed_x2": SteeringRecipe("target", False, 2),
    "smoothed_x4": SteeringRecipe("target", False, 4),
}


def preprocess_camera(image):
    """Convert a simulator frame to DeZero's pretrained float32 BGR format."""
    return np.ascontiguousarray(VGG16.preprocess(image, size=None))


def steering_loss_weights(rows, sharp_multiplier=1.0, sharp_threshold=0.3, *, downweight_spikes=True):
    """Boost large original commands and optionally apply indexed spike weights."""
    if not np.isfinite(sharp_multiplier) or sharp_multiplier < 1:
        raise ValueError("sharp_multiplier must be finite and at least 1")
    if not np.isfinite(sharp_threshold) or not 0 < sharp_threshold <= 1:
        raise ValueError("sharp_threshold must be in (0, 1]")
    weights = np.ones(len(rows), dtype=np.float32)
    if downweight_spikes:
        weights[:] = [row["weight"] for row in rows]
    sharp = np.array([abs(row["raw"]) >= sharp_threshold and not row["spike"]
                      for row in rows], dtype=bool)
    weights[sharp] *= sharp_multiplier
    return weights.reshape(-1, 1)


def read_split(index_path):
    """Read seeded sample rows or a legacy repository-relative index."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(index, list):
        return index
    if index["path_base"] != "repository_root":
        raise ValueError(f"Unsupported path base in {index_path}")
    samples = index["samples"]
    if len(samples) != index["sample_count"]:
        raise ValueError(f"Incorrect sample count in {index_path}")
    return [dict(sample, raw=sample["steering"], target=sample["steering"])
            for sample in samples]
