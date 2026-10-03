"""Fine-tunes a COCO-pretrained YOLO11 detector on the prepared waste data."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/taco/data.yaml")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="models/waste.pt")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.size, batch=args.batch,
                project=str(Path("runs").resolve()), name="waste", exist_ok=True)

    best = Path("runs") / "waste" / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
