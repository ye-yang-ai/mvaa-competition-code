#!/usr/bin/env python3
"""Prepare Task3 labeled videos as MedSAM2 video propagation inputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK3_DIR = REPO_ROOT / "task3"
if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))

from dataset import read_binary_mask_from_label_tar  # noqa: E402


IMG_RE = re.compile(r"^(?P<video>.+)_(?P<idx>\d{6})\.png$")


@dataclass(frozen=True)
class LabeledFrame:
    video_id: str
    frame_idx: int
    image_path: Path
    label_tar_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/pseudo/task3_medsam2/medsam2_inputs")
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--window-after", type=int, default=10, help="Number of labeled frames after each seed.")
    parser.add_argument("--window-before", type=int, default=0, help="Number of labeled frames before each seed.")
    parser.add_argument("--seed-stride", type=int, default=8, help="Use every Nth valid seed within each video.")
    parser.add_argument("--max-videos", type=int, default=2, help="Debug limit; 0 means all videos.")
    parser.add_argument("--max-seeds-per-video", type=int, default=2, help="Debug limit; 0 means all seeds.")
    parser.add_argument("--min-seed-fg-ratio", type=float, default=0.001)
    parser.add_argument("--max-seed-fg-ratio", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    return parser.parse_args()


def discover_labeled_frames(video_dir: Path) -> list[LabeledFrame]:
    frames: list[LabeledFrame] = []
    for image_path in sorted(video_dir.glob("*.png")):
        m = IMG_RE.match(image_path.name)
        if not m:
            continue
        video_id = m.group("video")
        frame_idx = int(m.group("idx"))
        label_tar = image_path.with_name(f"{video_id}_{m.group('idx')}_png_Label.tar")
        if label_tar.exists():
            frames.append(
                LabeledFrame(
                    video_id=video_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    label_tar_path=label_tar,
                )
            )
    return frames


def read_mask_stats(label_tar_path: Path, target_label: int) -> tuple[np.ndarray, dict[str, float]]:
    mask = read_binary_mask_from_label_tar(label_tar_path, target_label=target_label)
    mask_u8 = (mask > 0).astype(np.uint8)
    return mask_u8, {
        "foreground_pixels": float(mask_u8.sum()),
        "foreground_ratio": float(mask_u8.mean()),
    }


def save_frame_as_jpg(src_png: Path, dst_jpg: Path, jpeg_quality: int) -> None:
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src_png).convert("RGB").save(dst_jpg, quality=int(jpeg_quality))


def save_mask(mask_u8: np.ndarray, dst_png: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8.astype(np.uint8)).save(dst_png)


def candidate_seed_positions(
    frames: list[LabeledFrame],
    target_label: int,
    min_fg_ratio: float,
    max_fg_ratio: float,
    seed_stride: int,
    max_seeds: int,
) -> list[tuple[int, np.ndarray, dict[str, float]]]:
    selected: list[tuple[int, np.ndarray, dict[str, float]]] = []
    stride = max(1, int(seed_stride))
    for pos, frame in enumerate(frames):
        if pos % stride != 0:
            continue
        mask, stats = read_mask_stats(frame.label_tar_path, target_label=target_label)
        fg_ratio = stats["foreground_ratio"]
        if fg_ratio < float(min_fg_ratio) or fg_ratio > float(max_fg_ratio):
            continue
        selected.append((pos, mask, stats))
        if max_seeds > 0 and len(selected) >= max_seeds:
            break
    return selected


def prepare_one_seed_clip(
    frames: list[LabeledFrame],
    seed_pos: int,
    seed_mask: np.ndarray,
    seed_stats: dict[str, float],
    output_root: Path,
    target_label: int,
    window_before: int,
    window_after: int,
    jpeg_quality: int,
) -> dict:
    seed = frames[seed_pos]
    start = max(0, seed_pos - int(window_before))
    end = min(len(frames), seed_pos + int(window_after) + 1)
    clip = frames[start:end]
    clip_name = f"{seed.video_id}__seed_{seed.frame_idx:06d}"

    frames_dir = output_root / "videos" / clip_name
    masks_dir = output_root / "masks" / clip_name
    output_dir = output_root / "raw_outputs" / clip_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    seed_order = seed_pos - start
    for order, frame in enumerate(clip):
        frame_name = f"{order:05d}"
        save_frame_as_jpg(frame.image_path, frames_dir / f"{frame_name}.jpg", jpeg_quality=jpeg_quality)
        frame_records.append(
            {
                "order": order,
                "frame_name": frame_name,
                "source_video_id": frame.video_id,
                "source_frame_idx": frame.frame_idx,
                "image_path": str(frame.image_path),
                "label_tar_path": str(frame.label_tar_path),
            }
        )

    save_mask(seed_mask, masks_dir / f"{seed_order:05d}.png")

    return {
        "clip_name": clip_name,
        "source_video_id": seed.video_id,
        "seed_source_frame_idx": seed.frame_idx,
        "seed_order": seed_order,
        "target_label": int(target_label),
        "seed_mask": str(masks_dir / f"{seed_order:05d}.png"),
        "seed_mask_stats": seed_stats,
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "output_dir": str(output_dir),
        "num_frames": len(frame_records),
        "frames": frame_records,
    }


def main() -> int:
    args = parse_args()
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    video_dirs = [p for p in sorted(args.labeled_root.iterdir()) if p.is_dir()]
    if args.max_videos > 0:
        video_dirs = video_dirs[: args.max_videos]

    clips = []
    skipped_videos = []
    for video_dir in video_dirs:
        frames = discover_labeled_frames(video_dir)
        if not frames:
            skipped_videos.append({"video_dir": str(video_dir), "reason": "no_labeled_frames"})
            continue
        seeds = candidate_seed_positions(
            frames=frames,
            target_label=args.target_label,
            min_fg_ratio=args.min_seed_fg_ratio,
            max_fg_ratio=args.max_seed_fg_ratio,
            seed_stride=args.seed_stride,
            max_seeds=args.max_seeds_per_video,
        )
        if not seeds:
            skipped_videos.append({"video_dir": str(video_dir), "reason": "no_valid_seed"})
            continue
        for seed_pos, seed_mask, seed_stats in seeds:
            clips.append(
                prepare_one_seed_clip(
                    frames=frames,
                    seed_pos=seed_pos,
                    seed_mask=seed_mask,
                    seed_stats=seed_stats,
                    output_root=args.output_root,
                    target_label=args.target_label,
                    window_before=args.window_before,
                    window_after=args.window_after,
                    jpeg_quality=args.jpeg_quality,
                )
            )

    video_list_path = args.output_root / "video_list.txt"
    with video_list_path.open("w", encoding="utf-8") as f:
        for clip in clips:
            f.write(f"{clip['clip_name']}\n")

    manifest = {
        "labeled_root": str(args.labeled_root),
        "output_root": str(args.output_root),
        "target_label": int(args.target_label),
        "window_before": int(args.window_before),
        "window_after": int(args.window_after),
        "seed_stride": int(args.seed_stride),
        "min_seed_fg_ratio": float(args.min_seed_fg_ratio),
        "max_seed_fg_ratio": float(args.max_seed_fg_ratio),
        "video_list_file": str(video_list_path),
        "num_clips": len(clips),
        "clips": clips,
        "skipped_videos": skipped_videos,
    }
    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    rows = []
    for clip in clips:
        rows.append(
            {
                "clip_name": clip["clip_name"],
                "source_video_id": clip["source_video_id"],
                "seed_source_frame_idx": clip["seed_source_frame_idx"],
                "seed_order": clip["seed_order"],
                "num_frames": clip["num_frames"],
                "seed_foreground_pixels": clip["seed_mask_stats"]["foreground_pixels"],
                "seed_foreground_ratio": clip["seed_mask_stats"]["foreground_ratio"],
            }
        )
    csv_path = args.output_root / "seed_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"manifest": str(manifest_path), "seed_summary": str(csv_path), "num_clips": len(clips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
