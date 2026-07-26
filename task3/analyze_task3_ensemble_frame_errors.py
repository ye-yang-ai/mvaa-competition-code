#!/usr/bin/env python3
"""Frame-level diagnostics for a two-model Task3 probability ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from analyze_task3_frame_errors import (  # noqa: E402
    classify_frame,
    component_stats,
    dice_score,
    safe_float,
    top_rows,
)
from dataset import LabeledDataset, discover_samples, sample_has_foreground, split_train_val_by_video  # noqa: E402
from evaluate_task3_ensemble import build_model, pick_device  # noqa: E402
from evaluate_task3_postprocess_grid import (  # noqa: E402
    GridConfig,
    build_label_cache,
    cached_surface_distances,
    compute_metrics,
    postprocess_binary,
)
from train import predict_probs  # noqa: E402
from utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
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
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--keep-top", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true", default=False)
    parser.add_argument("--open-iters", type=int, default=0)
    parser.add_argument("--close-iters", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.nonzero(mask.astype(bool))
    if ys.size == 0:
        return None, None
    return float(ys.mean()), float(xs.mean())


def add_component_overlap_stats(pred: np.ndarray, label: np.ndarray, row: dict) -> None:
    labeled, n = ndi.label(pred.astype(bool))
    if n == 0:
        row.update(
            {
                "components_no_gt_overlap": 0,
                "components_no_gt_overlap_area": 0,
                "largest_no_gt_overlap_area": 0,
            }
        )
        return

    no_overlap = 0
    no_overlap_area = 0
    largest_no_overlap = 0
    for comp_id in range(1, int(n) + 1):
        comp = labeled == comp_id
        area = int(comp.sum())
        if not bool(np.logical_and(comp, label).any()):
            no_overlap += 1
            no_overlap_area += area
            largest_no_overlap = max(largest_no_overlap, area)
    row.update(
        {
            "components_no_gt_overlap": int(no_overlap),
            "components_no_gt_overlap_area": int(no_overlap_area),
            "largest_no_gt_overlap_area": int(largest_no_overlap),
        }
    )


@torch.no_grad()
def collect_ensemble_probs(args: argparse.Namespace, device: torch.device) -> dict:
    model_a, train_args_a = build_model(args.ckpt_a, device=device)
    model_b, train_args_b = build_model(args.ckpt_b, device=device)
    size_a = tuple(int(v) for v in train_args_a.get("image_size", [448, 800]))
    size_b = tuple(int(v) for v in train_args_b.get("image_size", [448, 800]))

    all_samples = discover_samples(args.labeled_root)
    _, val_samples, train_videos, val_videos = split_train_val_by_video(
        all_samples,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )
    if bool(args.val_only_fg):
        val_samples = [
            s for s in val_samples if sample_has_foreground(s, target_label=int(args.target_label))
        ]

    ds = LabeledDataset(
        samples=val_samples,
        image_size=size_a,
        target_label=int(args.target_label),
        train=False,
        cache_masks=True,
        use_imagenet_norm=bool(train_args_a.get("use_imagenet_norm", True)),
        seed=int(args.seed),
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    probs_list = []
    labels_list = []
    video_ids: list[str] = []
    frame_idxs: list[int] = []
    image_paths: list[str] = []
    use_amp = bool(args.amp and device.type == "cuda")
    weight_a = float(args.weight_a)
    weight_b = 1.0 - weight_a

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float()
        h, w = tuple(int(v) for v in labels.shape[-2:])

        x_a = images
        x_b = F.interpolate(images, size=size_b, mode="bilinear", align_corners=False)
        pa = predict_probs(model_a, x_a, use_amp=use_amp, use_tta=bool(args.tta))
        pb = predict_probs(model_b, x_b, use_amp=use_amp, use_tta=bool(args.tta))
        if tuple(pa.shape[-2:]) != (h, w):
            pa = F.interpolate(pa.float(), size=(h, w), mode="bilinear", align_corners=False)
        if tuple(pb.shape[-2:]) != (h, w):
            pb = F.interpolate(pb.float(), size=(h, w), mode="bilinear", align_corners=False)
        probs = weight_a * pa + weight_b * pb
        probs_list.append(probs.float().cpu().numpy()[:, 0])
        labels_list.append(labels.numpy()[:, 0])
        video_ids.extend([str(v) for v in batch["video_id"]])
        frame_idxs.extend([int(v) for v in batch["frame_idx"]])
        image_paths.extend([str(v) for v in batch["image_path"]])
        print(f"[collect] batch {batch_idx}/{len(loader)}")

    return {
        "probs": np.concatenate(probs_list, axis=0).astype(np.float32),
        "labels": np.concatenate(labels_list, axis=0).astype(bool),
        "video_ids": video_ids,
        "frame_idxs": frame_idxs,
        "image_paths": image_paths,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "train_args_a": train_args_a,
        "train_args_b": train_args_b,
        "size_a": size_a,
        "size_b": size_b,
    }


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    collected = collect_ensemble_probs(args, device)
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
    prev_pred_area_by_video: dict[str, int] = {}
    prev_gt_area_by_video: dict[str, int] = {}
    prev_pred_centroid_by_video: dict[str, tuple[float | None, float | None]] = {}
    prev_gt_centroid_by_video: dict[str, tuple[float | None, float | None]] = {}

    for i, (pred, label) in enumerate(zip(preds, labels)):
        video_id = collected["video_ids"][i]
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
        pred_cy, pred_cx = centroid(pred)
        gt_cy, gt_cx = centroid(label)
        prev_pred_area = prev_pred_area_by_video.get(video_id)
        prev_gt_area = prev_gt_area_by_video.get(video_id)
        prev_pred_c = prev_pred_centroid_by_video.get(video_id)
        prev_gt_c = prev_gt_centroid_by_video.get(video_id)
        pred_centroid_shift = None
        gt_centroid_shift = None
        if pred_cy is not None and prev_pred_c is not None and prev_pred_c[0] is not None:
            pred_centroid_shift = math.hypot(pred_cy - float(prev_pred_c[0]), pred_cx - float(prev_pred_c[1]))
        if gt_cy is not None and prev_gt_c is not None and prev_gt_c[0] is not None:
            gt_centroid_shift = math.hypot(gt_cy - float(prev_gt_c[0]), gt_cx - float(prev_gt_c[1]))

        prob = probs[i]
        row = {
            "video_id": video_id,
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
            "pred_area_delta_prev": None if prev_pred_area is None else int(pred_area - prev_pred_area),
            "gt_area_delta_prev": None if prev_gt_area is None else int(gt_area - prev_gt_area),
            "pred_centroid_shift_prev": safe_float(pred_centroid_shift) if pred_centroid_shift is not None else None,
            "gt_centroid_shift_prev": safe_float(gt_centroid_shift) if gt_centroid_shift is not None else None,
            "inter_area": inter,
            "fp_area": fp_area,
            "fn_area": fn_area,
            "union_area": union,
            "prob_mean": float(prob.mean()),
            "prob_max": float(prob.max()),
            "prob_p95": float(np.quantile(prob, 0.95)),
            "prob_p99": float(np.quantile(prob, 0.99)),
            **component_stats(pred),
        }
        add_component_overlap_stats(pred, label, row)
        rows.append(row)
        prev_pred_area_by_video[video_id] = pred_area
        prev_gt_area_by_video[video_id] = gt_area
        prev_pred_centroid_by_video[video_id] = (pred_cy, pred_cx)
        prev_gt_centroid_by_video[video_id] = (gt_cy, gt_cx)

    rows_by_hd = top_rows(rows, "hd", int(args.top_k), reverse=True)
    rows_by_asd = top_rows(rows, "asd", int(args.top_k), reverse=True)
    rows_by_low_dice = top_rows(rows, "dice", int(args.top_k), reverse=False)
    rows_by_no_overlap_area = top_rows(rows, "components_no_gt_overlap_area", int(args.top_k), reverse=True)
    rows_by_pred_shift = top_rows(rows, "pred_centroid_shift_prev", int(args.top_k), reverse=True)
    rows_empty_fp = sorted(
        [r for r in rows if r["status"] == "empty_fp"],
        key=lambda r: (r["pred_area"], r["prob_max"]),
        reverse=True,
    )[: int(args.top_k)]

    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_csv(args.output_dir / "worst_hd.csv", rows_by_hd)
    write_csv(args.output_dir / "worst_asd.csv", rows_by_asd)
    write_csv(args.output_dir / "lowest_dice.csv", rows_by_low_dice)
    write_csv(args.output_dir / "worst_no_gt_overlap_components.csv", rows_by_no_overlap_area)
    write_csv(args.output_dir / "largest_pred_centroid_shifts.csv", rows_by_pred_shift)
    write_csv(args.output_dir / "empty_false_positives.csv", rows_empty_fp)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    summary = {
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
        "weight_a": float(args.weight_a),
        "weight_b": float(1.0 - float(args.weight_a)),
        "config": cfg.__dict__,
        "labeled_root": str(args.labeled_root),
        "train_videos": collected["train_videos"],
        "val_videos": collected["val_videos"],
        "val_only_fg": bool(args.val_only_fg),
        "num_frames": len(rows),
        "aggregate": aggregate,
        "status_counts": status_counts,
        "top_worst_hd": rows_by_hd[:5],
        "top_lowest_dice": rows_by_low_dice[:5],
        "top_no_gt_overlap_components": rows_by_no_overlap_area[:5],
        "top_pred_centroid_shifts": rows_by_pred_shift[:5],
        "top_empty_false_positives": rows_empty_fp[:5],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "status_counts": status_counts,
                "top_worst_hd": rows_by_hd[:5],
                "top_no_gt_overlap_components": rows_by_no_overlap_area[:5],
                "top_empty_false_positives": rows_empty_fp[:5],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
