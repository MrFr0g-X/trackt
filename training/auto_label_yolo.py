from __future__ import annotations
import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
from tqdm import tqdm

CLASS_TARGET_STANDING = 0
CLASS_TARGET_FALLEN = 1
CLASS_RC_CAR = 2
MAX_TARGETS_PER_FRAME = 6
MAX_TARGETS_PER_COLOR = 3


@dataclass(frozen=True)
class Label:
    cls: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class FallBox:
    frame: int
    box: tuple[int, int, int, int]


def _controlled_masks(hsv: np.ndarray) -> list[np.ndarray]:
    red_low = cv2.inRange(
        hsv, np.array((0, 75, 55), np.uint8), np.array((13, 255, 255), np.uint8)
    )
    red_high = cv2.inRange(
        hsv, np.array((166, 75, 55), np.uint8), np.array((179, 255, 255), np.uint8)
    )
    red = cv2.bitwise_or(red_low, red_high)
    blue = cv2.inRange(
        hsv, np.array((90, 65, 40), np.uint8), np.array((130, 255, 255), np.uint8)
    )
    yellow = cv2.inRange(
        hsv, np.array((16, 95, 80), np.uint8), np.array((42, 255, 255), np.uint8)
    )
    return [red, blue, yellow]


def _contour_boxes(
    mask: np.ndarray,
    min_area: float,
    *,
    min_fill: float = 0.12,
) -> list[tuple[int, int, int, int, float]]:
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, int, int, float]] = []
    h, w = mask.shape[:2]
    frame_area = w * h
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 10 or bh < 10:
            continue
        box_area = bw * bh
        fill = area / max(1, box_area)
        if fill < min_fill:
            continue
        if box_area > frame_area * 0.12:
            continue
        out.append((x, y, bw, bh, area))
    out.sort(key=lambda b: b[4], reverse=True)
    return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def _center_distance(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))


def _dedupe(labels: Iterable[Label]) -> list[Label]:
    out: list[Label] = []
    for label in sorted(labels, key=lambda l: l.box[2] * l.box[3], reverse=True):
        if any(
            label.cls == kept.cls and _iou(label.box, kept.box) > 0.35 for kept in out
        ):
            continue
        out.append(label)
    return out


def _load_fall_boxes(repo: Path) -> dict[str, list[FallBox]]:
    out: dict[str, list[FallBox]] = {}
    eval_dir = repo / "test_videos" / "trackt_eval"
    for path in eval_dir.glob("*.result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        boxes = []
        for ev in data.get("fall_events", {}).values():
            boxes.append(
                FallBox(
                    frame=int(ev["frame"]),
                    box=(
                        int(ev["box_x"]),
                        int(ev["box_y"]),
                        int(ev["box_w"]),
                        int(ev["box_h"]),
                    ),
                )
            )
        out[path.name.replace(".result.json", "")] = boxes
    return out


def _is_fallen(
    frame: int, box: tuple[int, int, int, int], fall_boxes: list[FallBox]
) -> bool:
    x, y, w, h = box
    ratio = h / max(1, w)
    shape_fallen = ratio < 1.05 or w > h * 1.25
    if shape_fallen:
        return True
    cx_box = (x + w / 2, y + h / 2)
    for fb in fall_boxes:
        if frame < fb.frame - 8:
            continue
        if (
            _iou(box, fb.box) > 0.08
            or _center_distance(box, fb.box) < max(w, h, 80) * 1.8
        ):
            return True
    return False


def _target_labels(
    img: np.ndarray, frame: int, fall_boxes: list[FallBox]
) -> list[Label]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    min_area = max(3200.0, w * h * 0.00045)
    candidates: list[tuple[float, Label]] = []
    for mask in _controlled_masks(hsv):
        color_candidates: list[tuple[float, Label]] = []
        for x, y, bw, bh, area in _contour_boxes(mask, min_area, min_fill=0.16):
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            if bw < 22 or bh < 22 or max(bw, bh) < 46:
                continue
            if bw > w * 0.34 or bh > h * 0.62:
                continue
            ratio = bh / max(1, bw)
            if ratio > 5.5 or ratio < 0.22:
                continue
            if bw * bh < min_area * 1.15:
                continue
            cls = (
                CLASS_TARGET_FALLEN
                if _is_fallen(frame, (x, y, bw, bh), fall_boxes)
                else CLASS_TARGET_STANDING
            )
            shape_bonus = 1.0
            if cls == CLASS_TARGET_STANDING and 1.0 <= ratio <= 4.8:
                shape_bonus = 1.35
            elif cls == CLASS_TARGET_FALLEN and 0.22 <= ratio <= 1.25:
                shape_bonus = 1.20
            color_candidates.append((area * shape_bonus, Label(cls, (x, y, bw, bh))))
        color_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates.extend(color_candidates[:MAX_TARGETS_PER_COLOR])
    candidates.sort(key=lambda item: item[0], reverse=True)
    return _dedupe([lab for _, lab in candidates])[:MAX_TARGETS_PER_FRAME]


def _car_labels(img: np.ndarray, target_labels: list[Label]) -> list[Label]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    hh, ss, vv = cv2.split(hsv)
    dark = ((vv < 105) & (ss > 15)).astype(np.uint8) * 255
    blue = cv2.inRange(
        hsv, np.array((88, 60, 25), np.uint8), np.array((133, 255, 220), np.uint8)
    )
    mask = cv2.bitwise_or(dark, blue)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    min_area = max(9000.0, w * h * 0.0012)
    candidates = []
    for x, y, bw, bh, area in _contour_boxes(mask, min_area, min_fill=0.08):
        box = (x, y, bw, bh)
        if bw < 55 or bh < 45:
            continue
        if bw > w * 0.50 or bh > h * 0.55:
            continue
        if any(
            _iou(box, t.box) > 0.20 and bw * bh < t.box[2] * t.box[3] * 2.2
            for t in target_labels
        ):
            continue
        score = area * (1.0 + min(2.0, bw / max(1, bh)) * 0.1)
        candidates.append((score, box))
    candidates.sort(reverse=True)
    return [Label(CLASS_RC_CAR, candidates[0][1])] if candidates else []


def _filter_targets_against_cars(
    targets: list[Label], cars: list[Label]
) -> list[Label]:
    if not cars:
        return targets
    out: list[Label] = []
    for target in targets:
        tx, ty, tw, th = target.box
        tcx, tcy = tx + tw / 2, ty + th / 2
        drop = False
        for car in cars:
            cx, cy, cw, ch = car.box
            expanded = (cx - cw * 0.08, cy - ch * 0.08, cw * 1.16, ch * 1.16)
            inside = (
                expanded[0] <= tcx <= expanded[0] + expanded[2]
                and expanded[1] <= tcy <= expanded[1] + expanded[3]
            )
            if inside or _iou(target.box, car.box) > 0.08:
                drop = True
                break
        if not drop:
            out.append(target)
    return out


def _resize_and_write(
    img: np.ndarray, labels: list[Label], out_img: Path, out_txt: Path, width: int
) -> None:
    h, w = img.shape[:2]
    scale = width / w if width and w > width else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (width, round(h * scale)), interpolation=cv2.INTER_AREA)
    oh, ow = img.shape[:2]
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    for lab in labels:
        x, y, bw, bh = lab.box
        x *= scale
        y *= scale
        bw *= scale
        bh *= scale
        cx = (x + bw / 2) / ow
        cy = (y + bh / 2) / oh
        nw = bw / ow
        nh = bh / oh
        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{lab.cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _split_for_video(video: str) -> str:
    if "20260503_000708" in video:
        return "val"
    if "20260503_000448" in video:
        return "test"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="..")
    parser.add_argument("--frames", default="datasets/trackt_frames")
    parser.add_argument("--out", default="datasets/trackt_yolo")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    frames_root = Path(args.frames).resolve()
    out_root = Path(args.out).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    fall_boxes = _load_fall_boxes(repo)
    manifest = frames_root / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats = {
        "images": 0,
        "target_standing": 0,
        "target_fallen": 0,
        "rc_car": 0,
        "empty": 0,
    }
    for rec in tqdm(records, desc="auto-label"):
        img_path = frames_root / rec["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        video_stem = Path(rec["video"]).stem
        frame = int(rec["frame"])
        target_labels = _target_labels(img, frame, fall_boxes.get(video_stem, []))
        car_labels = _car_labels(img, target_labels)
        target_labels = _filter_targets_against_cars(target_labels, car_labels)
        labels = _dedupe([*target_labels, *car_labels])
        split = _split_for_video(rec["video"])
        name = img_path.name
        out_img = out_root / "images" / split / name
        out_txt = out_root / "labels" / split / f"{img_path.stem}.txt"
        _resize_and_write(img, labels, out_img, out_txt, args.width)
        stats["images"] += 1
        if not labels:
            stats["empty"] += 1
        for lab in labels:
            if lab.cls == CLASS_TARGET_STANDING:
                stats["target_standing"] += 1
            elif lab.cls == CLASS_TARGET_FALLEN:
                stats["target_fallen"] += 1
            elif lab.cls == CLASS_RC_CAR:
                stats["rc_car"] += 1
    (out_root / "label_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
