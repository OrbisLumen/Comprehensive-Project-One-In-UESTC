"""Export and identify the validation winner of a steering experiment."""

import json
from pathlib import Path
from shutil import copyfile


def export_best_checkpoint(weights_dir, recipe):
    """Copy the winning recipe checkpoint into the run's best.npz."""
    best = weights_dir / "best.npz"
    temporary = best.with_suffix(".partial.npz")
    copyfile(weights_dir / f"{recipe}.npz", temporary)
    temporary.replace(best)
    return best


def checkpoint_recipe(checkpoint, recipe=None):
    """Resolve a named checkpoint or read the winner from its run report."""
    checkpoint = Path(checkpoint).resolve()
    name = recipe or checkpoint.stem
    if name != "best":
        return name
    models_dir = checkpoint.parent.parent
    report = models_dir.parent.parent / "runs/donkeycar" / checkpoint.parent.name / "metrics.json"
    if models_dir.parts[-2:] == ("models", "donkeycar") and report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))["selected_recipe"]
    raise ValueError("For a standalone best.npz, pass recipe= with the selected training recipe.")
