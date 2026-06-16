#!/usr/bin/env python3
"""Filter MedSAM2 Task3 raw masks and create review overlays."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs/pseudo/task3_medsam2/medsam2_inputs/manifest.json",
    )
    parser.add_argument("--raw-mask-root", type=Path, default=REPO_ROOT / "outputs/pseudo/task3_medsam2/raw_masks_debug")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/pseudo/task3_medsam2/filtered_masks_debug")
    parser.add_argument("--min-fg-ratio", type=float, default=0.0002)
    parser.add_argument("--max-fg-ratio", type=float, default=0.25)
    parser.add_argument(
        "--max-seed-fg-ratio",
        type=float,
        default=1.0,
        help="Reject a whole propagated clip when the seed mask foreground ratio is larger than this.",
    )
    parser.add_argument("--max-seed-area-ratio", type=float, default=4.0)
    parser.add_argument("--min-seed-area-ratio", type=float, default=0.02)
    parser.add_argument("--max-adjacent-area-ratio", type=float, default=4.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path), dtype=np.uint8) > 0).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def overlay_mask(image_path: Path, mask: np.ndarray, save_path: Path, alpha: float) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    out = image.copy()
    fg = mask > 0
    color = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    out[fg] = (1.0 - alpha) * out[fg] + alpha * color
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(save_path, quality=95)


def area_ratio(num: int, den: int) -> float:
    if den <= 0:
        return float("inf") if num > 0 else 1.0
    return float(num) / float(den)


def decide_keep(
    fg_ratio: float,
    fg_pixels: int,
    seed_fg_pixels: int,
    prev_fg_pixels: int | None,
    min_fg_ratio: float,
    max_fg_ratio: float,
    min_seed_area_ratio: float,
    max_seed_area_ratio: float,
    max_adjacent_area_ratio: float,
) -> tuple[bool, str]:
    if fg_pixels <= 0:
        return False, "empty"
    if fg_ratio < min_fg_ratio:
        return False, "too_small"
    if fg_ratio > max_fg_ratio:
        return False, "too_large"

    rel_seed = area_ratio(fg_pixels, seed_fg_pixels)
    if rel_seed < min_seed_area_ratio:
        return False, "too_small_vs_seed"
    if rel_seed > max_seed_area_ratio:
        return False, "too_large_vs_seed"

    if prev_fg_pixels is not None and prev_fg_pixels > 0:
        rel_prev = area_ratio(max(fg_pixels, prev_fg_pixels), min(fg_pixels, prev_fg_pixels))
        if rel_prev > max_adjacent_area_ratio:
            return False, "adjacent_area_jump"

    return True, "keep"


def main() -> int:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    filtered_root = args.output_root / "masks"
    overlay_root = args.output_root / "overlays"
    reports_root = args.output_root / "reports"
    filtered_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    rows = []
    kept = 0
    rejected = 0
    reason_counts: dict[str, int] = {}

    for clip in manifest["clips"]:
        clip_name = clip["clip_name"]
        seed_fg_pixels = int(round(float(clip["seed_mask_stats"]["foreground_pixels"])))
        seed_fg_ratio = float(clip["seed_mask_stats"].get("foreground_ratio", 0.0))
        reject_clip_reason = "seed_too_large" if seed_fg_ratio > float(args.max_seed_fg_ratio) else ""
        prev_fg_pixels: int | None = None
        for frame in clip["frames"]:
            frame_name = frame["frame_name"]
            raw_mask_path = args.raw_mask_root / clip_name / f"{frame_name}.png"
            if reject_clip_reason:
                keep = False
                reason = reject_clip_reason
                fg_pixels = 0
                fg_ratio = 0.0
                mask = None
            elif not raw_mask_path.exists():
                keep = False
                reason = "missing_raw_mask"
                fg_pixels = 0
                fg_ratio = 0.0
                mask = None
            else:
                mask = read_mask(raw_mask_path)
                fg_pixels = int(mask.sum())
                fg_ratio = float(mask.mean())
                keep, reason = decide_keep(
                    fg_ratio=fg_ratio,
                    fg_pixels=fg_pixels,
                    seed_fg_pixels=seed_fg_pixels,
                    prev_fg_pixels=prev_fg_pixels,
                    min_fg_ratio=args.min_fg_ratio,
                    max_fg_ratio=args.max_fg_ratio,
                    min_seed_area_ratio=args.min_seed_area_ratio,
                    max_seed_area_ratio=args.max_seed_area_ratio,
                    max_adjacent_area_ratio=args.max_adjacent_area_ratio,
                )
                prev_fg_pixels = fg_pixels

            source_frame_idx = int(frame["source_frame_idx"])
            source_video_id = str(frame["source_video_id"])
            pseudo_name = f"{source_video_id}_{source_frame_idx:06d}_label_bin.png"
            filtered_path = filtered_root / source_video_id / pseudo_name
            overlay_path = overlay_root / clip_name / f"{frame_name}_overlay.jpg"

            if keep and mask is not None:
                save_binary_mask(mask, filtered_path)
                overlay_mask(Path(frame["image_path"]), mask, overlay_path, alpha=args.overlay_alpha)
                kept += 1
            else:
                rejected += 1
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            rows.append(
                {
                    "clip_name": clip_name,
                    "source_video_id": source_video_id,
                    "source_frame_idx": source_frame_idx,
                    "frame_name": frame_name,
                    "raw_mask_path": str(raw_mask_path),
                    "filtered_mask_path": str(filtered_path) if keep else "",
                    "overlay_path": str(overlay_path) if keep else "",
                    "seed_source_frame_idx": clip["seed_source_frame_idx"],
                    "seed_foreground_pixels": seed_fg_pixels,
                    "seed_foreground_ratio": seed_fg_ratio,
                    "foreground_pixels": fg_pixels,
                    "foreground_ratio": fg_ratio,
                    "keep": int(keep),
                    "reason": reason,
                }
            )

    csv_path = reports_root / "filter_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(args.manifest),
        "raw_mask_root": str(args.raw_mask_root),
        "output_root": str(args.output_root),
        "total_masks": len(rows),
        "kept": kept,
        "rejected": rejected,
        "reason_counts": reason_counts,
        "filter_report": str(csv_path),
        "filtered_root": str(filtered_root),
        "overlay_root": str(overlay_root),
    }
    summary_path = reports_root / "filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
