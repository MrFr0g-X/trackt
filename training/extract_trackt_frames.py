from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
from tqdm import tqdm

DEFAULT_VIDEOS = [
    "test_videos/20260503_000150.mp4",
    "test_videos/20260503_000315.mp4",
    "test_videos/20260503_000448.mp4",
    "test_videos/20260503_000708.mp4",
]


@dataclass
class FrameRecord:
    image: str
    video: str
    frame: int
    t_sec: float
    reason: str


def _fall_frames(repo: Path, stem: str, fps: float, radius_sec: float) -> set[int]:
    result = repo / "test_videos" / "trackt_eval" / f"{stem}.result.json"
    if not result.exists():
        return set()
    data = json.loads(result.read_text(encoding="utf-8"))
    frames: set[int] = set()
    radius = max(1, round(radius_sec * fps))
    for raw_frame in data.get("fall_events", {}).values():
        center = int(raw_frame.get("frame", 0))
        for frame in range(max(0, center - radius), center + radius + 1, 3):
            frames.add(frame)
    return frames


def extract_video(
    repo: Path,
    video_path: Path,
    out_dir: Path,
    base_stride: int,
    path_stride: int,
    fall_radius_sec: float,
) -> list[FrameRecord]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stem = video_path.stem
    fall_frames = _fall_frames(repo, stem, fps, fall_radius_sec)
    records: list[FrameRecord] = []
    selected: dict[int, str] = {}
    for frame in range(0, total, max(1, base_stride)):
        selected[frame] = "base"
    for frame in range(0, total, max(1, path_stride)):
        selected.setdefault(frame, "path")
    for frame in fall_frames:
        if frame < total:
            selected[frame] = "fall_window"
    image_dir = out_dir / "images_unlabeled"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_frames = set(selected)
    with tqdm(total=total, desc=stem) as bar:
        frame = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if frame in selected_frames:
                name = f"{stem}_f{frame:06d}.jpg"
                path = image_dir / name
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                records.append(
                    FrameRecord(
                        image=str(path.relative_to(out_dir)).replace("\\", "/"),
                        video=str(video_path.relative_to(repo)).replace("\\", "/"),
                        frame=frame,
                        t_sec=frame / fps,
                        reason=selected[frame],
                    )
                )
            frame += 1
            bar.update(1)
    cap.release()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="..", help="Repo root from training/")
    parser.add_argument("--out", default="datasets/trackt_frames")
    parser.add_argument("--base-stride", type=int, default=45)
    parser.add_argument("--path-stride", type=int, default=12)
    parser.add_argument("--fall-radius-sec", type=float, default=1.5)
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[FrameRecord] = []
    for rel in args.videos:
        all_records.extend(
            extract_video(
                repo=repo,
                video_path=(repo / rel).resolve(),
                out_dir=out_dir,
                base_stride=args.base_stride,
                path_stride=args.path_stride,
                fall_radius_sec=args.fall_radius_sec,
            )
        )
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    print(f"Extracted {len(all_records)} frames")
    print(f"Images:   {out_dir / 'images_unlabeled'}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
