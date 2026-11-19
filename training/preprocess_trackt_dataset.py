from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

TARGET_RANGES = {
    "red": [(0, 14, 65, 45), (164, 179, 65, 45)],
    "blue": [(88, 133, 55, 35)],
    "yellow": [(15, 42, 90, 70)],
    "green": [(43, 86, 55, 35)],
}
CANONICAL_BGR = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
}


def _mask_for_ranges(
    hsv: np.ndarray, ranges: list[tuple[int, int, int, int]]
) -> np.ndarray:
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h0, h1, s0, v0 in ranges:
        part = cv2.inRange(
            hsv, np.array((h0, s0, v0), np.uint8), np.array((h1, 255, 255), np.uint8)
        )
        out = cv2.bitwise_or(out, part)
    return out


def _target_mask(hsv: np.ndarray) -> np.ndarray:
    out = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in TARGET_RANGES.values():
        out = cv2.bitwise_or(out, _mask_for_ranges(hsv, ranges))
    return out


def apply_illumination_norm(bgr: np.ndarray, clip_limit: float = 2.4) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def enhance_targets(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = bgr.copy()
    for name, ranges in TARGET_RANGES.items():
        mask = _mask_for_ranges(hsv, ranges)
        out[mask > 0] = CANONICAL_BGR[name]
    return out


def enhance_car(bgr: np.ndarray, dark_value_cutoff: int = 120) -> np.ndarray:
    norm = apply_illumination_norm(bgr)
    dim = cv2.convertScaleAbs(norm, alpha=0.72, beta=-8)
    boosted = cv2.convertScaleAbs(norm, alpha=1.85, beta=42)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    dark = ((v <= dark_value_cutoff) & (s >= 10)).astype(np.uint8) * 255
    blue = ((h >= 86) & (h <= 134) & (s >= 35) & (v >= 15) & (v <= 235)).astype(
        np.uint8
    ) * 255
    car = cv2.bitwise_or(dark, blue)
    car = cv2.bitwise_and(car, cv2.bitwise_not(_target_mask(hsv)))
    car = cv2.morphologyEx(
        car, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    dim[car > 0] = boosted[car > 0]
    return dim


def _rewrite_labels_for_car_only(src: Path, dst: Path) -> int:
    kept = 0
    lines_out: list[str] = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            if parts[0] != "2":
                continue
            parts[0] = "0"
            lines_out.append(" ".join(parts))
            kept += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return kept


def build_dataset(
    src: Path, dst: Path, *, mode: str, car_only: bool, overwrite: bool
) -> None:
    if dst.exists():
        if not overwrite:
            raise SystemExit(f"Destination exists: {dst} (pass --overwrite)")
        shutil.rmtree(dst)
    stats = {"images": 0, "empty": 0, "rc_car": 0}
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        images = sorted((src / "images" / split).glob("*.*"))
        for img_path in tqdm(images, desc=f"{split}:{mode}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if mode == "car":
                out = enhance_car(img)
            elif mode == "target":
                out = enhance_targets(img)
            elif mode == "dual":
                out = enhance_car(enhance_targets(img))
            else:
                raise ValueError(mode)
            out_img = dst / "images" / split / img_path.name
            cv2.imwrite(str(out_img), out)
            src_label = src / "labels" / split / f"{img_path.stem}.txt"
            dst_label = dst / "labels" / split / f"{img_path.stem}.txt"
            if car_only:
                kept = _rewrite_labels_for_car_only(src_label, dst_label)
                stats["rc_car"] += kept
                if kept == 0:
                    stats["empty"] += 1
            else:
                if src_label.exists():
                    shutil.copy2(src_label, dst_label)
                else:
                    dst_label.write_text("", encoding="utf-8")
            stats["images"] += 1
    (dst / "preprocess_profile.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "car_only": car_only,
                "target_ranges": TARGET_RANGES,
                "stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/trackt_yolo_carplus")
    parser.add_argument("--dst", default="datasets/trackt_yolo_carplus_car_enhanced")
    parser.add_argument("--mode", choices=["car", "target", "dual"], default="car")
    parser.add_argument(
        "--car-only",
        action="store_true",
        help="Keep only rc_car labels and remap it to class 0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(
        Path(args.src),
        Path(args.dst),
        mode=args.mode,
        car_only=args.car_only,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
