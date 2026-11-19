from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from auto_label_yolo import (
    CLASS_RC_CAR,
    CLASS_TARGET_FALLEN,
    CLASS_TARGET_STANDING,
    Label,
    _dedupe,
    _filter_targets_against_cars,
    _resize_and_write,
    _target_labels,
)

CAR_CLASS_NAMES = {"car", "truck"}


def _hash_split(key: str) -> str:
    n = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 10
    if n == 0:
        return "val"
    if n == 1:
        return "test"
    return "train"


def _video_split(category: str, stem: str) -> str:
    val = {
        "car_only": {"20260505_203626"},
        "near_target": {"20260505_203206"},
        "collision": {"20260505_204659"},
    }
    test = {
        "car_only": {"20260505_203454"},
        "near_target": {"20260505_203009_1"},
        "collision": {"20260505_204350"},
    }
    if stem in val.get(category, set()):
        return "val"
    if stem in test.get(category, set()):
        return "test"
    return "train"


def _expand_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, int, int],
    frac: float = 0.035,
) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x, y, bw, bh = box
    pad_x = max(4, round(bw * frac))
    pad_y = max(4, round(bh * frac))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _teacher_car_label(
    model: YOLO,
    img: np.ndarray,
    *,
    min_conf: float,
    imgsz: int,
) -> Label | None:
    h, w = img.shape[:2]
    result = model.predict(img, imgsz=imgsz, conf=min_conf, verbose=False)[0]
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for box in result.boxes:
        cls = int(box.cls[0])
        name = str(model.names[cls])
        if name not in CAR_CLASS_NAMES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 28 or bh < 22:
            continue
        if bw > w * 0.60 or bh > h * 0.62:
            continue
        area = bw * bh
        score = conf * (1.0 + min(2.0, (area / max(1, w * h * 0.04))))
        candidates.append((score, (x1, y1, bw, bh)))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return Label(CLASS_RC_CAR, _expand_box(candidates[0][1], img.shape))


def _clean_new_target_labels(img: np.ndarray, labels: list[Label]) -> list[Label]:
    h, w = img.shape[:2]
    cleaned: list[Label] = []
    for label in labels:
        x, y, bw, bh = label.box
        bottom = y + bh
        center_y = y + bh / 2
        area = bw * bh
        if bottom < h * 0.26:
            continue
        if center_y < h * 0.22:
            continue
        if center_y < h * 0.32 and bh < h * 0.20:
            continue
        if label.cls == CLASS_TARGET_FALLEN and y < h * 0.18 and area < w * h * 0.012:
            continue
        if area < w * h * 0.00030:
            continue
        cleaned.append(label)
    return cleaned


def _copy_existing_dataset(src: Path, dst: Path) -> dict[str, int]:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    stats_path = dst / "label_stats.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        "images": 0,
        "target_standing": 0,
        "target_fallen": 0,
        "rc_car": 0,
        "empty": 0,
    }


def _add_image(
    *,
    img: np.ndarray,
    name: str,
    split: str,
    frame: int,
    out_root: Path,
    width: int,
    teacher: YOLO,
    teacher_conf: float,
    teacher_imgsz: int,
    stats: dict[str, int],
) -> bool:
    target_labels = _clean_new_target_labels(img, _target_labels(img, frame, []))
    car = _teacher_car_label(
        teacher,
        img,
        min_conf=teacher_conf,
        imgsz=teacher_imgsz,
    )
    car_labels = [car] if car is not None else []
    target_labels = _filter_targets_against_cars(target_labels, car_labels)
    labels = _dedupe([*target_labels, *car_labels])
    if not labels:
        return False
    out_img = out_root / "images" / split / f"{name}.jpg"
    out_txt = out_root / "labels" / split / f"{name}.txt"
    _resize_and_write(img, labels, out_img, out_txt, width)
    stats["images"] = stats.get("images", 0) + 1
    for lab in labels:
        if lab.cls == CLASS_TARGET_STANDING:
            stats["target_standing"] = stats.get("target_standing", 0) + 1
        elif lab.cls == CLASS_TARGET_FALLEN:
            stats["target_fallen"] = stats.get("target_fallen", 0) + 1
        elif lab.cls == CLASS_RC_CAR:
            stats["rc_car"] = stats.get("rc_car", 0) + 1
    return True


def _sample_video_frames(video: Path, stride: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected: list[tuple[int, np.ndarray]] = []
    for frame_idx in range(0, frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            selected.append((frame_idx, frame))
    cap.release()
    return selected


def _draw_review_tile(img: np.ndarray, label_file: Path) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    colors = {
        CLASS_TARGET_STANDING: (0, 210, 255),
        CLASS_TARGET_FALLEN: (0, 255, 80),
        CLASS_RC_CAR: (255, 95, 0),
    }
    names = {
        CLASS_TARGET_STANDING: "standing",
        CLASS_TARGET_FALLEN: "fallen",
        CLASS_RC_CAR: "rc_car",
    }
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls_s, cx_s, cy_s, bw_s, bh_s = line.split()
            cls = int(cls_s)
            cx, cy, bw, bh = map(float, (cx_s, cy_s, bw_s, bh_s))
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                names.get(cls, str(cls)),
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def _write_review_sheet(
    out_root: Path, review_path: Path, max_images: int = 40
) -> None:
    image_files = sorted((out_root / "images").glob("*/*carplus*.jpg"))
    if not image_files:
        return
    idxs = np.linspace(
        0, len(image_files) - 1, min(max_images, len(image_files))
    ).astype(int)
    tiles: list[np.ndarray] = []
    labels: list[str] = []
    for idx in idxs:
        img_path = image_files[int(idx)]
        label_path = out_root / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        tiles.append(_draw_review_tile(img, label_path))
        labels.append(f"{img_path.parent.name}/{img_path.name}")
    cols = 4
    cell_w, cell_h = 320, 210
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 24, dtype=np.uint8)
    for i, (img, label) in enumerate(zip(tiles, labels)):
        row, col = divmod(i, cols)
        x0, y0 = col * cell_w, row * cell_h
        h, w = img.shape[:2]
        scale = min(cell_w / w, (cell_h - 30) / h)
        resized = cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
        y = y0 + ((cell_h - 30) - resized.shape[0]) // 2
        x = x0 + (cell_w - resized.shape[1]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            label[:42],
            (x0 + 8, y0 + cell_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="..")
    parser.add_argument("--base", default="datasets/trackt_yolo")
    parser.add_argument("--out", default="datasets/trackt_yolo_carplus")
    parser.add_argument("--teacher", default="yolo11n.pt")
    parser.add_argument("--teacher-conf", type=float, default=0.05)
    parser.add_argument("--teacher-imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--clear-stride", type=int, default=1)
    parser.add_argument("--video-stride", type=int, default=24)
    parser.add_argument(
        "--review", default="../test_videos/car_data_review/carplus_label_review.jpg"
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    base = (
        Path(args.base) if Path(args.base).is_absolute() else Path.cwd() / args.base
    ).resolve()
    out_root = (
        Path(args.out) if Path(args.out).is_absolute() else Path.cwd() / args.out
    ).resolve()
    stats = _copy_existing_dataset(base, out_root)
    added = {
        "clear_frames": 0,
        "car_only": 0,
        "near_target": 0,
        "collision": 0,
    }
    teacher_path = Path(args.teacher)
    if not teacher_path.is_absolute():
        teacher_path = Path.cwd() / teacher_path
    teacher = YOLO(str(teacher_path))
    clear_dir = repo / "test_videos" / "clear frames of the RC car"
    clear_images = sorted(
        p for p in clear_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for index, path in enumerate(
        tqdm(clear_images[:: args.clear_stride], desc="clear car frames")
    ):
        img = cv2.imread(str(path))
        if img is None:
            continue
        split = _hash_split(path.stem)
        ok = _add_image(
            img=img,
            name=f"carplus_clear_{path.stem}",
            split=split,
            frame=index,
            out_root=out_root,
            width=args.width,
            teacher=teacher,
            teacher_conf=args.teacher_conf,
            teacher_imgsz=args.teacher_imgsz,
            stats=stats,
        )
        added["clear_frames"] += int(ok)
    video_sources = [
        ("car_only", repo / "test_videos" / "Car-only videos"),
        ("near_target", repo / "test_videos" / "Car-near-target videos"),
        ("collision", repo / "test_videos" / "Collision videos"),
    ]
    for category, folder in video_sources:
        videos = sorted(folder.glob("*.mp4"))
        for video in videos:
            split = _video_split(category, video.stem)
            frames = _sample_video_frames(video, args.video_stride)
            for frame_idx, img in tqdm(
                frames, desc=f"{category}:{video.stem}", leave=False
            ):
                ok = _add_image(
                    img=img,
                    name=f"carplus_{category}_{video.stem}_f{frame_idx:06d}",
                    split=split,
                    frame=frame_idx,
                    out_root=out_root,
                    width=args.width,
                    teacher=teacher,
                    teacher_conf=args.teacher_conf,
                    teacher_imgsz=args.teacher_imgsz,
                    stats=stats,
                )
                added[category] += int(ok)
    stats["added_carplus"] = added
    (out_root / "label_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    _write_review_sheet(
        out_root,
        (
            Path(args.review)
            if Path(args.review).is_absolute()
            else Path.cwd() / args.review
        ),
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
