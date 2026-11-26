import json

import numpy as np
import pytest

from donkeycar_app.checkpoints import checkpoint_recipe, export_best_checkpoint
from donkeycar_app.dataset import STEERING_RECIPES


def test_best_is_an_exact_copy_and_resolves_to_its_recipe(tmp_path):
    weights = tmp_path / "models/donkeycar/run-1"
    run = tmp_path / "runs/donkeycar/run-1"
    weights.mkdir(parents=True)
    run.mkdir(parents=True)
    for i, recipe in enumerate(STEERING_RECIPES):
        np.savez(weights / f"{recipe}.npz", value=i)

    for winner in ("raw", "downweighted_x4"):
        best = export_best_checkpoint(weights, winner)
        (run / "metrics.json").write_text(json.dumps({"selected_recipe": winner}), encoding="utf-8")
        assert checkpoint_recipe(best) == winner
        assert best.read_bytes() == (weights / f"{winner}.npz").read_bytes()
        assert {p.stem for p in weights.iterdir()} == {*STEERING_RECIPES, "best"}


def test_standalone_and_legacy_checkpoints(tmp_path):
    best = tmp_path / "best.npz"
    with pytest.raises(ValueError, match="recipe="):
        checkpoint_recipe(best)
    assert checkpoint_recipe(best, "raw_x4") == "raw_x4"
    assert checkpoint_recipe(tmp_path / "interpolated.npz") == "interpolated"
