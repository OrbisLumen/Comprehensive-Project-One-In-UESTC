"""Build split indices from archived simulator drives and repair training spikes."""

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]


def read_drive(folder, metadata, training):
    rows = []
    for path in folder.glob("record_*.json"):
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        image = folder / record["cam/image_array"]
        angle = float(record["user/angle"])
        throttle = float(record["user/throttle"])
        if not (-1 <= angle <= 1 and math.isfinite(throttle)):
            raise ValueError(f"Invalid controls in {path}")
        rows.append(dict(route=folder.name, seed=metadata["seed"],
                         frame=int(path.stem.split("_")[-1]),
                         image=image.relative_to(ROOT).as_posix(),
                         record=path.relative_to(ROOT).as_posix(), throttle=throttle,
                         raw=angle, target=angle, weight=1.0, spike=False, sustained=False))
    if not rows or len(rows) != metadata["frames"]:
        raise ValueError(f"Incomplete recording: {folder}")
    rows.sort(key=lambda row: row["frame"])
    for before, current, after in zip(rows, rows[1:], rows[2:]):
        if (before["frame"] + 1 != current["frame"] or current["frame"] + 1 != after["frame"]
                or min(before["throttle"], current["throttle"], after["throttle"]) <= 0):
            continue
        a, b, c = before["raw"], current["raw"], after["raw"]
        current["spike"] = abs(b) == 1 and a*b < 0 and b*c < 0
        current["sustained"] = min(abs(a), abs(b), abs(c)) >= 0.3 and a*b > 0 and b*c > 0
    if training:
        for before, current, after in zip(rows, rows[1:], rows[2:]):
            if current["spike"]:
                current["weight"] = 0.1
                if not before["spike"] and not after["spike"]:
                    current["target"] = (before["raw"] + after["raw"]) / 2
    return rows


def main():
    config = json.loads((ROOT / "examples/donkeycar/road-seed-splits.json").read_text("utf-8"))
    archive = ROOT / config["raw_dir"]
    output = ROOT / config["index_dir"]
    indices = {}
    layout_split = {}
    image_split = {}
    seen_seeds = set()
    for split in ("train", "validation", "test"):
        part = []
        for seed in config[split]:
            if seed in seen_seeds:
                raise ValueError(f"Repeated seed {seed}")
            seen_seeds.add(seed)
            manifests = sorted(archive.glob(f"seed-{seed}-*/collection.json"))
            if not manifests:
                raise FileNotFoundError(f"No recording for seed {seed} under {archive}")
            for path in manifests:
                metadata = json.loads(path.read_text("utf-8"))
                if metadata["seed"] != seed:
                    raise ValueError(f"Seed mismatch in {path}")
                if layout_split.setdefault(metadata["layout_hash"], split) != split:
                    raise ValueError(f"Road layout shared across splits: {path}")
                rows = read_drive(path.parent, metadata, split == "train")
                for row in rows:
                    digest = hashlib.sha256((ROOT / row["image"]).read_bytes()).digest()
                    if image_split.setdefault(digest, split) != split:
                        raise ValueError(f"Image shared across splits: {row['image']}")
                part.extend(rows)
        indices[split] = part
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        for split, rows in indices.items():
            path = Path(tmp) / f"{split}.json"
            path.write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
        for split, rows in indices.items():
            name = f"{split}.json"
            (Path(tmp) / name).replace(output / name)
            print(split, len(rows), "frames,", sum(r["raw"] != r["target"] for r in rows),
                  "repaired,", sum(r["weight"] < 1 for r in rows), "downweighted")


if __name__ == "__main__":
    main()
