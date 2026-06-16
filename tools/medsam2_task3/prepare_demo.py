#!/usr/bin/env python3
"""Prepare a tiny Task3 video clip for the official MedSAM2 video demo."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK3_DIR = REPO_ROOT / "task3"
if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))

from dataset import read_binary_mask_from_label_tar  # noqa: E402


IMG_RE = re.compile(r"^(?P<video>.+)_(?P<idx>\d{6})\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=REPO_ROOT / "data/reference_data/t3_vid/train/REC_20250109_110158_583A",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/pseudo/task3_medsam2/demo_official",
    )
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0, help="Use the Nth labeled frame as seed.")
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    return parser.parse_args()


def discover_labeled_frames(video_dir: Path) -> list[tuple[int, Path, Path]]:
    out: list[tuple[int, Path, Path]] = []
    for image_path in sorted(video_dir.glob("*.png")):
        m = IMG_RE.match(image_path.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        label_tar = image_path.with_name(f"{m.group('video')}_{m.group('idx')}_png_Label.tar")
        if label_tar.exists():
            out.append((idx, image_path, label_tar))
    if not out:
        raise RuntimeError(f"No labeled frames found in {video_dir}")
    return out


def save_frame_as_jpg(src_png: Path, dst_jpg: Path) -> None:
    img = Image.open(src_png).convert("RGB")
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_jpg, quality=95)


def save_mask_png(label_tar: Path, dst_png: Path, target_label: int) -> dict[str, float]:
    mask = read_binary_mask_from_label_tar(label_tar, target_label=target_label)
    mask_u8 = (mask > 0).astype(np.uint8)
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8).save(dst_png)
    return {
        "foreground_pixels": float(mask_u8.sum()),
        "foreground_ratio": float(mask_u8.mean()),
    }


def main() -> int:
    args = parse_args()
    labeled = discover_labeled_frames(args.video_dir)
    seed_pos = min(max(int(args.seed_offset), 0), len(labeled) - 1)
    clip = labeled[seed_pos : seed_pos + int(args.num_frames)]
    if not clip:
        raise RuntimeError("Empty demo clip after applying seed offset and num frames.")

    video_name = args.video_dir.name
    frames_dir = args.output_root / "videos" / video_name
    masks_dir = args.output_root / "masks" / video_name
    output_dir = args.output_root / "outputs"

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for order, (frame_idx, image_path, label_tar) in enumerate(clip):
        frame_name = f"{order:05d}"
        save_frame_as_jpg(image_path, frames_dir / f"{frame_name}.jpg")
        frame_records.append(
            {
                "order": order,
                "frame_name": frame_name,
                "frame_idx": frame_idx,
                "image_path": str(image_path),
                "label_tar": str(label_tar),
            }
        )

    seed_idx, _, seed_label_tar = clip[0]
    mask_stats = save_mask_png(seed_label_tar, masks_dir / "00000.png", target_label=args.target_label)

    manifest = {
        "video_name": video_name,
        "video_dir": str(args.video_dir),
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "output_dir": str(output_dir),
        "seed_frame_idx": seed_idx,
        "target_label": int(args.target_label),
        "seed_mask": str(masks_dir / "00000.png"),
        "seed_mask_stats": mask_stats,
        "frames": frame_records,
    }
    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
