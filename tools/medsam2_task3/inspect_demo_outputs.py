#!/usr/bin/env python3
"""Inspect MedSAM2 demo masks and create simple overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=REPO_ROOT / "outputs/pseudo/task3_medsam2/demo_official",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    out = rgb.copy()
    fg = mask > 0
    color = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    out[fg] = (1.0 - alpha) * out[fg] + alpha * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main() -> int:
    args = parse_args()
    manifest_path = args.demo_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    video_name = manifest["video_name"]
    frames_dir = args.demo_root / "videos" / video_name
    masks_dir = args.demo_root / "outputs" / video_name
    reports_dir = args.demo_root / "reports"
    overlays_dir = reports_dir / "overlays"
    reports_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in manifest["frames"]:
        frame_name = item["frame_name"]
        image_path = frames_dir / f"{frame_name}.jpg"
        mask_path = masks_dir / f"{frame_name}.png"
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
        fg_pixels = int((mask > 0).sum())
        fg_ratio = float((mask > 0).mean())
        overlay = overlay_mask(image, mask, alpha=args.alpha)
        overlay_path = overlays_dir / f"{frame_name}_overlay.jpg"
        overlay.save(overlay_path, quality=95)
        rows.append(
            {
                "frame_name": frame_name,
                "source_frame_idx": item["frame_idx"],
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "width": image.width,
                "height": image.height,
                "foreground_pixels": fg_pixels,
                "foreground_ratio": fg_ratio,
            }
        )

    csv_path = reports_dir / "mask_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "video_name": video_name,
        "num_masks": len(rows),
        "mean_foreground_ratio": float(np.mean([r["foreground_ratio"] for r in rows])),
        "min_foreground_ratio": float(np.min([r["foreground_ratio"] for r in rows])),
        "max_foreground_ratio": float(np.max([r["foreground_ratio"] for r in rows])),
        "csv_path": str(csv_path),
        "overlays_dir": str(overlays_dir),
    }
    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
