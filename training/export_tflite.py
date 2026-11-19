from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", default="trackt_dataset.yaml")
    parser.add_argument("--flutter-assets", default="../flutter_app/assets/models")
    args = parser.parse_args()
    model = YOLO(args.weights)
    exported = model.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=args.int8,
        nms=False,
        data=args.data,
    )
    exported_path = Path(exported)
    assets = Path(args.flutter_assets).resolve()
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, assets / "detect.tflite")
    names = model.names
    labels = [
        str(names[i] if isinstance(names, dict) else names[i])
        for i in range(len(names))
    ]
    (assets / "labelmap.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"Copied model to {assets / 'detect.tflite'}")
    print(f"Wrote labels to {assets / 'labelmap.txt'}")


if __name__ == "__main__":
    main()
