#!/usr/bin/env python3
"""Frame-level Task3 error diagnostics for a checkpoint/config."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from evaluate_task3_postprocess_grid import (
    GridConfig,
    build_label_cache,
    build_model,
    cached_surface_distances,
    collect_validation_probs,
    compute_metrics,
    pick_device,
    postprocess_binary,
)
from dataset import discover_samples, split_train_val_by_video
from utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt"),
    )
    parser.add_argument("--labeled-root", type=Path, default=Path("data/reference_data/t3_vid/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--threshold", type=float, default=0.285)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--keep-top", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true", default=False)
    parser.add_argument("--open-iters", type=int, default=0)
    parser.add_argument("--close-iters", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def safe_float(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    denom = int(pred.sum()) + int(target.sum())
    return float((2.0 * inter + 1e-6) / (denom + 1e-6))


def component_stats(mask: np.ndarray) -> dict:
    labeled, n = ndi.label(mask.astype(bool))
    if n == 0:
        return {
            "pred_components": 0,
            "largest_component_area": 0,
            "top3_component_areas": "",
            "small_components_lt20": 0,
            "small_components_lt50": 0,
            "small_components_lt100": 0,
        }
    areas = np.bincount(labeled.ravel())[1:]
    areas_sorted = sorted((int(x) for x in areas), reverse=True)
    return {
        "pred_components": int(n),
        "largest_component_area": int(areas_sorted[0]),
        "top3_component_areas": ",".join(str(x) for x in areas_sorted[:3]),
        "small_components_lt20": int(sum(x < 20 for x in areas)),
        "small_components_lt50": int(sum(x < 50 for x in areas)),
        "small_components_lt100": int(sum(x < 100 for x in areas)),
    }


def classify_frame(gt_area: int, pred_area: int) -> str:
    if gt_area == 0 and pred_area == 0:
        return "empty_tn"
    if gt_area == 0 and pred_area > 0:
        return "empty_fp"
    if gt_area > 0 and pred_area == 0:
        return "fg_miss"
    return "fg_hit"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_rows(rows: list[dict], key: str, k: int, reverse: bool = True) -> list[dict]:
    valid = [r for r in rows if r.get(key) is not None]
    return sorted(valid, key=lambda r: r[key], reverse=reverse)[:k]


@torch.no_grad()
def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model, train_args, ckpt = build_model(args.ckpt_path, device=device)

    all_samples = discover_samples(args.labeled_root)
    _, val_samples, train_videos, val_videos = split_train_val_by_video(
        all_samples,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )
    print(
        f"[setup] device={device} val_frames={len(val_samples)} val_videos={val_videos} "
        f"thr={args.threshold} min_area={args.min_area}"
    )

    collected = collect_validation_probs(
        model=model,
        train_args=train_args,
        samples=val_samples,
        args=args,
        device=device,
    )
    probs = collected["probs"]
    labels = collected["labels"].astype(bool)
    label_cache = build_label_cache(labels)
    cfg = GridConfig(
        threshold=float(args.threshold),
        min_area=int(args.min_area),
        keep_top=int(args.keep_top),
        fill_holes=bool(args.fill_holes),
        open_iters=int(args.open_iters),
        close_iters=int(args.close_iters),
    )

    raw = probs > float(args.threshold)
    preds = np.zeros_like(raw, dtype=bool)
    for i in range(raw.shape[0]):
        preds[i] = postprocess_binary(raw[i], cfg)

    aggregate = compute_metrics(
        preds,
        label_cache,
        collected["video_ids"],
        include_per_video=True,
        include_distance=True,
    )

    rows: list[dict] = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        pred_area = int(pred.sum())
        gt_area = int(label.sum())
        inter = int(np.logical_and(pred, label).sum())
        fp_area = int(np.logical_and(pred, ~label).sum())
        fn_area = int(np.logical_and(~pred, label).sum())
        union = int(np.logical_or(pred, label).sum())
        dist = cached_surface_distances(pred, label_cache, i)
        hd = None
        asd = None
        if dist is not None:
            hd, asd = dist
        comp = component_stats(pred)
        prob = probs[i]
        row = {
            "video_id": collected["video_ids"][i],
            "frame_idx": int(collected["frame_idxs"][i]),
            "image_path": collected["image_paths"][i],
            "case_id": Path(collected["image_paths"][i]).stem,
            "status": classify_frame(gt_area, pred_area),
            "dice": dice_score(pred, label),
            "hd": safe_float(hd) if hd is not None else None,
            "asd": safe_float(asd) if asd is not None else None,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "area_ratio_pred_gt": safe_float(pred_area / gt_area) if gt_area > 0 else None,
            "inter_area": inter,
            "fp_area": fp_area,
            "fn_area": fn_area,
            "union_area": union,
            "prob_mean": float(prob.mean()),
            "prob_max": float(prob.max()),
            "prob_p95": float(np.quantile(prob, 0.95)),
            "prob_p99": float(np.quantile(prob, 0.99)),
            **comp,
        }
        rows.append(row)

    rows_by_hd = top_rows(rows, "hd", int(args.top_k), reverse=True)
    rows_by_asd = top_rows(rows, "asd", int(args.top_k), reverse=True)
    rows_by_low_dice = top_rows(rows, "dice", int(args.top_k), reverse=False)
    rows_empty_fp = sorted(
        [r for r in rows if r["status"] == "empty_fp"],
        key=lambda r: (r["pred_area"], r["prob_max"]),
        reverse=True,
    )[: int(args.top_k)]

    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_csv(args.output_dir / "worst_hd.csv", rows_by_hd)
    write_csv(args.output_dir / "worst_asd.csv", rows_by_asd)
    write_csv(args.output_dir / "lowest_dice.csv", rows_by_low_dice)
    write_csv(args.output_dir / "empty_false_positives.csv", rows_empty_fp)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    summary = {
        "ckpt_path": str(args.ckpt_path),
        "train_args": train_args,
        "ckpt_val_metrics": ckpt.get("val_metrics", {}),
        "labeled_root": str(args.labeled_root),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "config": cfg.__dict__,
        "aggregate": aggregate,
        "status_counts": status_counts,
        "top_worst_hd": rows_by_hd,
        "top_lowest_dice": rows_by_low_dice,
        "top_empty_false_positives": rows_empty_fp,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "aggregate": aggregate,
        "status_counts": status_counts,
        "top_worst_hd": rows_by_hd[:5],
        "top_empty_false_positives": rows_empty_fp[:5],
    }, ensure_ascii=False, indent=2))
    print(f"[saved] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
