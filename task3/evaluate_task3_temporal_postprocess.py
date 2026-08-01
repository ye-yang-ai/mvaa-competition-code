#!/usr/bin/env python3
"""Evaluate temporal connected-component post-processing for Task3 ensembles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage as ndi

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from analyze_task3_ensemble_frame_errors import collect_ensemble_probs  # noqa: E402
from evaluate_task3_ensemble import pick_device  # noqa: E402
from evaluate_task3_postprocess_grid import build_label_cache, compute_metrics  # noqa: E402
from utils import MetricRefs, metric_quality_weighted, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, required=True)
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--val-only-fg", action="store_true", default=False)
    parser.add_argument("--no-val-only-fg", action="store_false", dest="val_only_fg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["raw", "largest", "temporal"])
    parser.add_argument("--min-extra-areas", type=int, nargs="+", default=[0, 20, 100, 400])
    parser.add_argument("--max-remove-areas", type=int, nargs="+", default=[50, 100, 400, 1000, 3000])
    parser.add_argument("--neighbor-dilate-iters", type=int, nargs="+", default=[3, 5, 8, 12])
    parser.add_argument("--min-overlap-pixels", type=int, nargs="+", default=[1, 20, 100])
    parser.add_argument("--dsc-weight", type=float, default=0.6)
    parser.add_argument("--hd-weight", type=float, default=0.2)
    parser.add_argument("--asd-weight", type=float, default=0.2)
    parser.add_argument("--hd-ref", type=float, default=20.0)
    parser.add_argument("--asd-ref", type=float, default=3.0)
    return parser.parse_args()


def main_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask.astype(bool))
    if n <= 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labeled.ravel())
    keep_id = int(np.argmax(areas[1:]) + 1)
    return labeled == keep_id


def dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    if int(iters) <= 0:
        return mask.astype(bool)
    return ndi.binary_dilation(
        mask.astype(bool),
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(iters),
    )


def temporal_postprocess_video(
    video_preds: np.ndarray,
    mode: str,
    min_extra_area: int,
    max_remove_area: int,
    neighbor_dilate_iters: int,
    min_overlap_pixels: int,
) -> np.ndarray:
    mode = str(mode)
    if mode == "raw":
        return video_preds.astype(bool)

    out = np.zeros_like(video_preds, dtype=bool)
    main_masks = [main_component(frame) for frame in video_preds]
    supported_regions: list[np.ndarray] = []
    for idx in range(len(main_masks)):
        support = np.zeros_like(main_masks[idx], dtype=bool)
        if idx > 0:
            support |= dilate(main_masks[idx - 1], int(neighbor_dilate_iters))
        if idx + 1 < len(main_masks):
            support |= dilate(main_masks[idx + 1], int(neighbor_dilate_iters))
        supported_regions.append(support)

    for idx, frame in enumerate(video_preds):
        labeled, n = ndi.label(frame.astype(bool))
        if n <= 0:
            continue
        areas = np.bincount(labeled.ravel())
        largest_id = int(np.argmax(areas[1:]) + 1)
        keep = np.zeros(n + 1, dtype=bool)
        keep[largest_id] = True

        if mode == "temporal":
            support = supported_regions[idx]
            for comp_id in range(1, int(n) + 1):
                if comp_id == largest_id:
                    continue
                area = int(areas[comp_id])
                if area < int(min_extra_area):
                    continue
                comp = labeled == comp_id
                if int(np.logical_and(comp, support).sum()) >= int(min_overlap_pixels):
                    keep[comp_id] = True
        elif mode == "remove_unsupported_small":
            support = supported_regions[idx]
            for comp_id in range(1, int(n) + 1):
                if comp_id == largest_id:
                    continue
                area = int(areas[comp_id])
                comp = labeled == comp_id
                has_support = int(np.logical_and(comp, support).sum()) >= int(min_overlap_pixels)
                if has_support or area > int(max_remove_area):
                    keep[comp_id] = True
        elif mode != "largest":
            raise ValueError(f"Unsupported mode: {mode}")

        out[idx] = keep[labeled]
    return out


def temporal_postprocess(
    preds: np.ndarray,
    video_ids: Sequence[str],
    frame_idxs: Sequence[int],
    mode: str,
    min_extra_area: int,
    max_remove_area: int,
    neighbor_dilate_iters: int,
    min_overlap_pixels: int,
) -> np.ndarray:
    out = np.zeros_like(preds, dtype=bool)
    for video_id in sorted(set(video_ids)):
        idxs = [i for i, v in enumerate(video_ids) if v == video_id]
        idxs = sorted(idxs, key=lambda i: int(frame_idxs[i]))
        video_preds = preds[idxs]
        pp = temporal_postprocess_video(
            video_preds,
            mode=mode,
            min_extra_area=int(min_extra_area),
            max_remove_area=int(max_remove_area),
            neighbor_dilate_iters=int(neighbor_dilate_iters),
            min_overlap_pixels=int(min_overlap_pixels),
        )
        for local_i, original_i in enumerate(idxs):
            out[original_i] = pp[local_i]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    collected = collect_ensemble_probs(args, device)
    probs = collected["probs"]
    labels = collected["labels"].astype(bool)
    label_cache = build_label_cache(labels)
    refs = MetricRefs(hd_ref=float(args.hd_ref), asd_ref=float(args.asd_ref))

    rows: list[dict] = []
    per_video_by_key: dict[str, list[dict]] = {}
    for threshold in args.thresholds:
        base_preds = probs > float(threshold)
        for mode in args.modes:
            if str(mode) in {"raw", "largest"}:
                combos = [(0, 0, 0, 0)]
            elif str(mode) == "remove_unsupported_small":
                combos = [
                    (0, int(max_remove_area), int(dilate_iters), int(min_overlap))
                    for max_remove_area in args.max_remove_areas
                    for dilate_iters in args.neighbor_dilate_iters
                    for min_overlap in args.min_overlap_pixels
                ]
            else:
                combos = [
                    (int(min_area), 0, int(dilate_iters), int(min_overlap))
                    for min_area in args.min_extra_areas
                    for dilate_iters in args.neighbor_dilate_iters
                    for min_overlap in args.min_overlap_pixels
                ]

            for min_extra_area, max_remove_area, dilate_iters, min_overlap in combos:
                preds = temporal_postprocess(
                    base_preds,
                    collected["video_ids"],
                    collected["frame_idxs"],
                    mode=str(mode),
                    min_extra_area=min_extra_area,
                    max_remove_area=max_remove_area,
                    neighbor_dilate_iters=dilate_iters,
                    min_overlap_pixels=min_overlap,
                )
                metrics = compute_metrics(
                    preds,
                    label_cache,
                    collected["video_ids"],
                    include_per_video=True,
                    include_distance=True,
                )
                score = metric_quality_weighted(
                    dsc=float(metrics["dice_all"]),
                    hd=float(metrics["hd"]),
                    asd=float(metrics["asd"]),
                    refs=refs,
                    dsc_weight=float(args.dsc_weight),
                    hd_weight=float(args.hd_weight),
                    asd_weight=float(args.asd_weight),
                )["score"]
                row = {
                    "threshold": float(threshold),
                    "mode": str(mode),
                    "min_extra_area": int(min_extra_area),
                    "max_remove_area": int(max_remove_area),
                    "neighbor_dilate_iters": int(dilate_iters),
                    "min_overlap_pixels": int(min_overlap),
                    "score": float(score),
                    "dice_all": float(metrics["dice_all"]),
                    "dice_fg": float(metrics["dice_fg"]),
                    "hd": float(metrics["hd"]),
                    "asd": float(metrics["asd"]),
                    "empty_fp_rate": float(metrics["empty_fp_rate"]),
                    "pred_pos_ratio": float(metrics["pred_pos_ratio"]),
                    "gt_pos_ratio": float(metrics["gt_pos_ratio"]),
                    "valid_dist_cases": int(metrics["valid_dist_cases"]),
                    "num_frames": int(metrics["num_frames"]),
                    "fg_frames": int(metrics["fg_frames"]),
                    "empty_frames": int(metrics["empty_frames"]),
                }
                rows.append(row)
                key = (
                    f"thr{float(threshold):.4f}_{mode}_area{min_extra_area}_"
                    f"maxrm{max_remove_area}_dil{dilate_iters}_ov{min_overlap}"
                )
                per_video_by_key[key] = metrics.get("per_video", [])

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    rows_by_hd = sorted(rows, key=lambda r: (r["hd"], -r["dice_all"]))
    rows_by_dice = sorted(rows, key=lambda r: r["dice_all"], reverse=True)
    raw_rows = [r for r in rows if r["mode"] == "raw"]
    raw_best = sorted(raw_rows, key=lambda r: r["score"], reverse=True)[0] if raw_rows else None

    write_csv(args.output_dir / "grid_results.csv", rows_sorted)
    with (args.output_dir / "grid_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, ensure_ascii=False, indent=2)
    result = {
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
        "weight_a": float(args.weight_a),
        "weight_b": float(1.0 - float(args.weight_a)),
        "val_videos": collected["val_videos"],
        "train_videos": collected["train_videos"],
        "num_val_samples": int(labels.shape[0]),
        "raw_best": raw_best,
        "best_by_score": rows_sorted[0],
        "best_by_hd": rows_by_hd[0],
        "best_by_dice": rows_by_dice[0],
        "top10_by_score": rows_sorted[:10],
        "top10_by_hd": rows_by_hd[:10],
        "grid": {
            "thresholds": [float(x) for x in args.thresholds],
            "modes": list(args.modes),
            "min_extra_areas": [int(x) for x in args.min_extra_areas],
            "max_remove_areas": [int(x) for x in args.max_remove_areas],
            "neighbor_dilate_iters": [int(x) for x in args.neighbor_dilate_iters],
            "min_overlap_pixels": [int(x) for x in args.min_overlap_pixels],
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "raw_best": raw_best,
        "best_by_score": rows_sorted[0],
        "best_by_hd": rows_by_hd[0],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
