"""Exports a YOLO detect model to TorchScript for the C++ engine.

The batch size is fixed at export time and must match the engine's --batch.
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/waste.pt")
    p.add_argument("--out", default="models/waste.torchscript")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.weights)
    # Trace on the GPU so tensors baked into the graph live on the right device.
    device = 0 if torch.cuda.is_available() else "cpu"
    path = model.export(format="torchscript", imgsz=args.size, batch=args.batch, device=device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes next to the weights, which may already be the target.
    if Path(path).resolve() != out.resolve():
        shutil.copy(path, out)
    names = model.names
    out.with_suffix(".names").write_text("\n".join(names[i] for i in sorted(names)) + "\n")
    print(f"saved {out} and {out.with_suffix('.names')} ({len(names)} classes, batch {args.batch})")


if __name__ == "__main__":
    main()
