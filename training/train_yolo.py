from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="trackt_dataset.yaml")
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/trackt")
    parser.add_argument("--name", default="trackt_yolo11n")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows to avoid CUDA DLL page-file failures.",
    )
    args = parser.parse_args()
    model = YOLO(args.base)
    result = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=20,
        plots=True,
        cache=False,
    )
    print(result)


if __name__ == "__main__":
    main()
