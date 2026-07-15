#!/usr/bin/env python3
"""Post-process Task3 binary mask PNGs for submission."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process Task3 submission masks.")
    parser.add_argument("--input-task-dir", type=Path, required=True)
    parser.add_argument("--output-task-dir", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=1200)
    parser.add_argument("--keep-top", type=int, default=4)
    parser.add_argument("--close-iters", type=int, default=1)
    parser.add_argument("--fill-holes", action="store_true", default=True)
    parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes")
    return parser.parse_args()


def postprocess_mask(
    mask: np.ndarray,
    min_area: int,
    keep_top: int,
    close_iters: int,
    fill_holes: bool,
) -> tuple[np.ndarray, dict]:
    before_area = int(mask.sum())
    labeled, num_components = ndi.label(mask)
    if num_components == 0:
        return mask.astype(bool), {
            "before_area": before_area,
            "after_area": 0,
            "components_before": 0,
            "components_after": 0,
            "kept_areas": [],
            "removed_area": before_area,
        }

    areas = np.bincount(labeled.ravel())[1:]
    order = np.argsort(areas)[::-1]
    keep_labels = []
    kept_areas = []
    for rank, comp_idx in enumerate(order):
        area = int(areas[comp_idx])
        if rank < keep_top and area >= min_area:
            keep_labels.append(comp_idx + 1)
            kept_areas.append(area)

    if not keep_labels:
        keep_labels = [int(order[0]) + 1]
        kept_areas = [int(areas[order[0]])]

    out = np.isin(labeled, keep_labels)
    if fill_holes:
        out = ndi.binary_fill_holes(out)
    if close_iters > 0:
        structure = np.ones((3, 3), dtype=bool)
        out = ndi.binary_closing(out, structure=structure, iterations=int(close_iters))

    labeled_after, components_after = ndi.label(out)
    after_area = int(out.sum())
    return out.astype(bool), {
        "before_area": before_area,
        "after_area": after_area,
        "components_before": int(num_components),
        "components_after": int(components_after),
        "kept_areas": kept_areas,
        "removed_area": int(max(0, before_area - after_area)),
    }


def main() -> int:
    args = parse_args()
    if not args.input_task_dir.is_dir():
        raise FileNotFoundError(f"Input Task3 directory not found: {args.input_task_dir}")

    args.output_task_dir.mkdir(parents=True, exist_ok=True)
    input_json = args.input_task_dir / "task3_predictions.json"
    output_json = args.output_task_dir / "task3_predictions.json"
    if input_json.exists():
        shutil.copy2(input_json, output_json)

    rows = []
    mask_paths = sorted(args.input_task_dir.rglob("*_label_bin.png"))
    if not mask_paths:
        raise RuntimeError(f"No Task3 mask PNGs found under: {args.input_task_dir}")

    for src in mask_paths:
        rel = src.relative_to(args.input_task_dir)
        dst = args.output_task_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        mask = np.asarray(Image.open(src).convert("L")) > 0
        out, stats = postprocess_mask(
            mask,
            min_area=args.min_area,
            keep_top=args.keep_top,
            close_iters=args.close_iters,
            fill_holes=args.fill_holes,
        )
        Image.fromarray((out.astype(np.uint8) * 255), mode="L").save(dst)
        rows.append({"path": rel.as_posix(), **stats})

    report = {
        "input_task_dir": str(args.input_task_dir),
        "output_task_dir": str(args.output_task_dir),
        "min_area": args.min_area,
        "keep_top": args.keep_top,
        "close_iters": args.close_iters,
        "fill_holes": args.fill_holes,
        "num_masks": len(mask_paths),
        "total_removed_area": int(sum(r["removed_area"] for r in rows)),
        "mean_components_before": float(np.mean([r["components_before"] for r in rows])),
        "mean_components_after": float(np.mean([r["components_after"] for r in rows])),
        "frames": rows,
    }
    with (args.output_task_dir / "postprocess_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Processed masks: {len(mask_paths)}")
    print(f"Saved to: {args.output_task_dir}")
    print(f"Total removed area: {report['total_removed_area']}")
    print(
        "Mean components: "
        f"{report['mean_components_before']:.2f} -> {report['mean_components_after']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
