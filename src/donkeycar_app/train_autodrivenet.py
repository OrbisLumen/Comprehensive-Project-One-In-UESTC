"""Fit the course CNN to the original controls on trace1 through trace5."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from donkeycar_app.autodrivenet import AutoDriveNet, hsv_frame


def read_split(root, routes):
    rows = []
    for route in routes:
        folder = root / "data/raw" / route
        records = sorted(folder.glob("record_*.json"), key=lambda p: int(p.stem[7:]))
        if not records:
            raise ValueError(f"No recorded controls in {folder}")
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            rows.append(dict(route=route, frame=int(path.stem[7:]),
                             image=folder / record["cam/image_array"], steering=record["user/angle"]))
    pixels = np.empty((len(rows), 120, 160, 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        path = row["image"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Unreadable camera image: {path}")
        pixels[i] = hsv_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    labels = torch.tensor([row["steering"] for row in rows], dtype=torch.float32)
    return rows, TensorDataset(torch.from_numpy(pixels), labels)


def batch_images(pixels, device):
    return pixels.to(device).permute(0, 3, 1, 2).float().div_(255)


@torch.inference_mode()
def predict(model, dataset, device, batch_size):
    model.eval()
    return torch.cat([
        model(batch_images(images, device)).cpu()
        for images, _ in DataLoader(dataset, batch_size=batch_size)
    ])


def evaluate(model, rows, dataset, device, batch_size, output):
    pred = predict(model, dataset, device, batch_size).clamp(-1, 1)
    targets = dataset.tensors[1]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("route", "frame", "target", "prediction"))
        for row, value in zip(rows, pred.tolist(), strict=True):
            writer.writerow((row["route"], row["frame"], row["steering"], value))
    residual = pred - targets
    sharp = targets.abs() >= 0.3
    return dict(frames=len(rows), mae=residual.abs().mean().item(),
                mse=residual.square().mean().item(), sharp_mae=residual[sharp].abs().mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.patience) < 1:
        parser.error("Batch size, epochs and patience must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    print(f"Reading routes on {device}", flush=True)
    _, train = read_split(args.root, [f"trace{i}" for i in range(1, 6)])
    val_rows, val = read_split(args.root, ["trace6"])
    train_batches = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    model = AutoDriveNet().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_mse = float("inf")
    best_epoch = 0
    checkpoint = args.out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        for pixels, targets in train_batches:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_images(pixels, device))
            loss = criterion(pred, targets.to(device))
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(targets)
        val_pred = predict(model, val, device, args.batch_size)
        val_mse = criterion(val_pred, val.tensors[1]).item()
        entry = dict(epoch=epoch, train_mse=total.item()/len(train), val_mse=val_mse)
        with (args.out / "epochs.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
        print(entry, flush=True)
        if val_mse < best_mse:
            best_epoch, best_mse = epoch, val_mse
            torch.save(model.state_dict(), checkpoint)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_rows, test = read_split(args.root, ["trace7", "trace8"])
    report = dict(best_epoch=best_epoch, epochs_completed=epoch, seed=42, lr=1e-4,
                  device=str(device), torch_version=str(torch.__version__), opencv=cv2.__version__,
                  batch_size=args.batch_size, patience=args.patience, max_epochs=args.epochs)
    for name, rows, dataset in [("validation", val_rows, val), ("test", test_rows, test)]:
        report[name] = evaluate(model, rows, dataset, device, args.batch_size,
                                args.out / f"{name}-predictions.csv")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
