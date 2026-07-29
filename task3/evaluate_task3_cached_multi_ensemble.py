#!/usr/bin/env python3
"""Cache N Task3 model probabilities and grid-search ensemble settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from dataset import LabeledDataset, discover_samples, split_train_val_by_video  # noqa: E402
from evaluate_task3_ensemble import build_model  # noqa: E402
from evaluate_task3_postprocess_grid import (  # noqa: E402
    GridConfig,
    build_label_cache,
    compute_metrics,
    postprocess_binary,
    write_csv,
)
from generate_task3_predictions import load_presence_gate  # noqa: E402
from train import predict_probs  # noqa: E402
from utils import MetricRefs, metric_quality_weighted, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", or "cuda:N"')
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--reuse-cache", action="store_true", default=False)
    parser.add_argument(
        "--presence-gate-ckpt",
        type=Path,
        default=None,
        help="Optional frame-level presence gate checkpoint. Applies only to selected model probability caches.",
    )
    parser.add_argument("--presence-gate-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--presence-gate-model-indices",
        type=int,
        nargs="*",
        default=[],
        help="0-based model indices gated by presence; defaults to the last model when a gate is provided.",
    )
    parser.add_argument("--presence-gate-tta", action="store_true", default=True)
    parser.add_argument("--no-presence-gate-tta", action="store_false", dest="presence_gate_tta")
    parser.add_argument(
        "--weight-grid",
        nargs="+",
        required=True,
        help="Semicolon-separated candidates per model, e.g. '0.25,0.30;0.15,0.20;...'",
    )
    parser.add_argument("--weight-sum-tol", type=float, default=1e-6)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.375, 0.40, 0.425])
    parser.add_argument("--min-areas", type=int, nargs="+", default=[0])
    parser.add_argument("--fill-holes", type=int, nargs="+", default=[0])
    parser.add_argument("--keep-tops", type=int, nargs="+", default=[0])
    parser.add_argument("--open-iters", type=int, nargs="+", default=[0])
    parser.add_argument("--close-iters", type=int, nargs="+", default=[0])
    parser.add_argument("--dsc-weight", type=float, default=0.6)
    parser.add_argument("--hd-weight", type=float, default=0.2)
    parser.add_argument("--asd-weight", type=float, default=0.2)
    parser.add_argument("--hd-ref", type=float, default=20.0)
    parser.add_argument("--asd-ref", type=float, default=3.0)
    parser.add_argument("--pareto-dice-drop", type=float, default=0.004)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--full-metric-top-k", type=int, default=400)
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    mode = str(device_arg).lower().strip()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda" or mode.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(mode)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_presence_gate_indices(args: argparse.Namespace, n_models: int) -> set[int]:
    if args.presence_gate_ckpt is None:
        return set()
    raw_indices = list(args.presence_gate_model_indices)
    if not raw_indices:
        raw_indices = [n_models - 1]
    indices = {int(i) for i in raw_indices}
    bad = sorted(i for i in indices if i < 0 or i >= n_models)
    if bad:
        raise ValueError(f"--presence-gate-model-indices out of range for {n_models} models: {bad}")
    return indices


def read_cache(cache_dir: Path, n_models: int) -> dict | None:
    required = [cache_dir / f"probs_{i}.npy" for i in range(n_models)]
    required.extend([cache_dir / "labels.npy", cache_dir / "metadata.json"])
    if not all(p.exists() for p in required):
        return None
    with (cache_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "probs": [np.load(cache_dir / f"probs_{i}.npy").astype(np.float32) for i in range(n_models)],
        "labels": np.load(cache_dir / "labels.npy").astype(bool),
        "metadata": metadata,
    }


@torch.no_grad()
def collect_model_probs(
    model: torch.nn.Module,
    train_args: dict,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
) -> np.ndarray:
    image_size = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    probs_list: list[np.ndarray] = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float()
        h, w = tuple(int(v) for v in labels.shape[-2:])
        x = F.interpolate(images, size=image_size, mode="bilinear", align_corners=False)
        probs = predict_probs(model, x, use_amp=use_amp, use_tta=use_tta)
        if tuple(probs.shape[-2:]) != (h, w):
            probs = F.interpolate(probs.float(), size=(h, w), mode="bilinear", align_corners=False)
        probs_list.append(probs.float().cpu().numpy()[:, 0])
        print(f"[collect] {batch_idx}/{len(loader)} size={image_size}")
    return np.concatenate(probs_list, axis=0).astype(np.float32)


@torch.no_grad()
def collect_presence_gate_probs(
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    gate_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    gate_model, gate_threshold, predict_fn = load_presence_gate(
        args.presence_gate_ckpt,
        device=device,
        threshold_override=float(args.presence_gate_threshold),
    )
    probs_list: list[np.ndarray] = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        x = F.interpolate(images, size=gate_size, mode="bilinear", align_corners=False)
        probs = predict_fn(
            gate_model,
            x,
            use_amp=use_amp,
            use_tta=bool(args.presence_gate_tta),
        )
        probs_list.append(probs.float().cpu().numpy())
        print(f"[presence-gate] {batch_idx}/{len(loader)} size={gate_size}")
    return np.concatenate(probs_list, axis=0).astype(np.float32), float(gate_threshold)


def build_cache(args: argparse.Namespace, device: torch.device) -> dict:
    cache_dir = args.output_dir / "cache"
    if bool(args.reuse_cache):
        cached = read_cache(cache_dir, n_models=len(args.ckpts))
        if cached is not None:
            print(f"[cache] reuse {cache_dir}")
            return cached

    all_samples = discover_samples(args.labeled_root)
    _, val_samples, train_videos, val_videos = split_train_val_by_video(
        all_samples,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )

    first_model, first_args = build_model(args.ckpts[0], device=device)
    models = [first_model]
    train_args_list = [first_args]
    for ckpt_path in args.ckpts[1:]:
        model, train_args = build_model(ckpt_path, device=device)
        models.append(model)
        train_args_list.append(train_args)

    base_size = tuple(int(v) for v in train_args_list[0].get("image_size", [448, 800]))
    ds = LabeledDataset(
        samples=val_samples,
        image_size=base_size,
        target_label=int(args.target_label),
        train=False,
        cache_masks=True,
        use_imagenet_norm=bool(train_args_list[0].get("use_imagenet_norm", True)),
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

    labels_list: list[np.ndarray] = []
    video_ids: list[str] = []
    frame_idxs: list[int] = []
    image_paths: list[str] = []
    for batch in loader:
        labels_list.append(batch["label"].numpy()[:, 0].astype(bool))
        video_ids.extend([str(v) for v in batch["video_id"]])
        frame_idxs.extend([int(v) for v in batch["frame_idx"]])
        image_paths.extend([str(v) for v in batch["image_path"]])
    labels = np.concatenate(labels_list, axis=0).astype(bool)

    use_amp = bool(args.amp and device.type == "cuda")
    probs = []
    for name, model, train_args in zip(args.names, models, train_args_list):
        print(f"[cache] collect {name}")
        probs.append(collect_model_probs(model, train_args, loader, device, use_amp=use_amp, use_tta=bool(args.tta)))

    gated_model_indices = resolve_presence_gate_indices(args, n_models=len(args.ckpts))
    presence_gate_metadata = None
    if args.presence_gate_ckpt is not None:
        first_gated = min(gated_model_indices)
        gate_size = tuple(int(v) for v in train_args_list[first_gated].get("image_size", [448, 800]))
        gate_probs, gate_threshold = collect_presence_gate_probs(
            args,
            loader=loader,
            device=device,
            use_amp=use_amp,
            gate_size=gate_size,
        )
        gate_keep = gate_probs >= float(gate_threshold)
        for model_idx in sorted(gated_model_indices):
            print(
                "[presence-gate] apply "
                f"model={args.names[model_idx]} idx={model_idx} clear={int((~gate_keep).sum())}/{len(gate_keep)}"
            )
            probs[model_idx] = probs[model_idx].copy()
            probs[model_idx][~gate_keep] = 0.0
        presence_gate_metadata = {
            "ckpt": str(args.presence_gate_ckpt),
            "threshold": float(gate_threshold),
            "model_indices": sorted(gated_model_indices),
            "model_names": [str(args.names[i]) for i in sorted(gated_model_indices)],
            "size": list(gate_size),
            "tta": bool(args.presence_gate_tta),
            "cleared_frames": int((~gate_keep).sum()),
            "kept_frames": int(gate_keep.sum()),
            "num_frames": int(len(gate_keep)),
        }

    metadata = {
        "ckpts": [str(p) for p in args.ckpts],
        "names": [str(n) for n in args.names],
        "train_videos": train_videos,
        "val_videos": val_videos,
        "num_val_samples": len(val_samples),
        "video_ids": video_ids,
        "frame_idxs": frame_idxs,
        "image_paths": image_paths,
        "base_size": list(base_size),
        "model_sizes": [list(tuple(int(v) for v in t.get("image_size", [448, 800]))) for t in train_args_list],
        "seed": int(args.seed),
        "val_video_count": int(args.val_video_count),
        "use_amp": use_amp,
        "use_tta": bool(args.tta),
        "presence_gate": presence_gate_metadata,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(probs):
        np.save(cache_dir / f"probs_{i}.npy", arr)
    np.save(cache_dir / "labels.npy", labels.astype(np.uint8))
    write_json(cache_dir / "metadata.json", metadata)
    return {"probs": probs, "labels": labels, "metadata": metadata}


def parse_weight_grid(values: Sequence[str], n_models: int) -> list[list[float]]:
    if len(values) == 1 and ";" in values[0]:
        parts = values[0].split(";")
    else:
        parts = list(values)
    if len(parts) != n_models:
        raise ValueError(f"--weight-grid must provide {n_models} groups, got {len(parts)}")
    grid: list[list[float]] = []
    for part in parts:
        vals = [round(float(v), 6) for v in part.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"Empty weight-grid group: {part!r}")
        grid.append(vals)
    return grid


def iter_weights(args: argparse.Namespace) -> list[tuple[float, ...]]:
    weight_grid = parse_weight_grid(args.weight_grid, n_models=len(args.ckpts))
    rows = []
    for weights in product(*weight_grid):
        s = float(sum(weights))
        if abs(s - 1.0) <= float(args.weight_sum_tol):
            rows.append(tuple(round(float(w), 6) for w in weights))
    if not rows:
        raise RuntimeError("No weight rows sum to 1.0; adjust --weight-grid or --weight-sum-tol.")
    return sorted(set(rows))


def iter_grid_configs(args: argparse.Namespace) -> list[GridConfig]:
    configs = []
    for threshold, min_area, fill_holes, keep_top, open_iters, close_iters in product(
        args.thresholds,
        args.min_areas,
        args.fill_holes,
        args.keep_tops,
        args.open_iters,
        args.close_iters,
    ):
        configs.append(
            GridConfig(
                threshold=float(threshold),
                min_area=int(min_area),
                keep_top=int(keep_top),
                fill_holes=bool(int(fill_holes)),
                open_iters=int(open_iters),
                close_iters=int(close_iters),
            )
        )
    return configs


def score_row(metrics: dict, args: argparse.Namespace) -> float:
    return float(
        metric_quality_weighted(
            dsc=float(metrics["dice_all"]),
            hd=float(metrics["hd"]),
            asd=float(metrics["asd"]),
            refs=MetricRefs(hd_ref=float(args.hd_ref), asd_ref=float(args.asd_ref)),
            dsc_weight=float(args.dsc_weight),
            hd_weight=float(args.hd_weight),
            asd_weight=float(args.asd_weight),
        )["score"]
    )


def evaluate_candidate(
    probs: np.ndarray,
    label_cache,
    video_ids: Sequence[str],
    cfg: GridConfig,
    args: argparse.Namespace,
    include_distance: bool,
) -> tuple[np.ndarray, dict]:
    raw = probs > float(cfg.threshold)
    if (
        int(cfg.min_area) <= 0
        and int(cfg.keep_top) <= 0
        and not bool(cfg.fill_holes)
        and int(cfg.open_iters) <= 0
        and int(cfg.close_iters) <= 0
    ):
        preds = raw.astype(bool)
    else:
        preds = np.zeros_like(raw, dtype=bool)
        for i in range(raw.shape[0]):
            preds[i] = postprocess_binary(raw[i], cfg)
    metrics = compute_metrics(
        preds,
        label_cache,
        video_ids,
        include_per_video=include_distance,
        include_distance=include_distance,
    )
    metrics["score"] = score_row(metrics, args) if include_distance else float(metrics["dice_all"])
    return preds, metrics


def candidate_row(weights: tuple[float, ...], cfg: GridConfig, metrics: dict, names: Sequence[str]) -> dict:
    row = {f"weight_{name}": float(weight) for name, weight in zip(names, weights)}
    row.update(
        {
            "threshold": float(cfg.threshold),
            "min_area": int(cfg.min_area),
            "fill_holes": int(cfg.fill_holes),
            "keep_top": int(cfg.keep_top),
            "open_iters": int(cfg.open_iters),
            "close_iters": int(cfg.close_iters),
            "score": float(metrics["score"]),
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
    )
    return row


def fast_score(row: dict) -> float:
    empty_fp = float(row["empty_fp_rate"])
    if math.isnan(empty_fp):
        empty_fp = 0.0
    gt_pos = max(float(row["gt_pos_ratio"]), 1e-8)
    pos_drift = abs(float(row["pred_pos_ratio"]) - float(row["gt_pos_ratio"])) / gt_pos
    return float(row["dice_all"]) - 0.02 * empty_fp - 0.01 * pos_drift


def row_key(row: dict, names: Sequence[str]) -> tuple:
    return tuple(
        [row[f"weight_{name}"] for name in names]
        + [
            row["threshold"],
            row["min_area"],
            row["fill_holes"],
            row["keep_top"],
            row["open_iters"],
            row["close_iters"],
        ]
    )


def select_full_metric_candidates(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    target = max(1, int(args.full_metric_top_k))
    best_dice = max(float(r["dice_all"]) for r in rows)
    dice_floor = best_dice - float(args.pareto_dice_drop)
    names = [str(n) for n in args.names]
    selected: dict[tuple, dict] = {}
    for row in sorted(rows, key=lambda r: r["dice_all"], reverse=True)[:target]:
        selected[row_key(row, names)] = row
    for row in sorted(rows, key=fast_score, reverse=True)[:target]:
        selected[row_key(row, names)] = row
    for row in rows:
        if float(row["dice_all"]) >= dice_floor:
            selected[row_key(row, names)] = row
        if len(selected) >= target * 3:
            break
    return list(selected.values())


def write_summary(args: argparse.Namespace, rows: list[dict], metadata: dict) -> None:
    rows_by_score = sorted(rows, key=lambda r: r["score"], reverse=True)
    best_dice = max(rows, key=lambda r: r["dice_all"])
    dice_floor = float(best_dice["dice_all"]) - float(args.pareto_dice_drop)
    near_best_dice = [r for r in rows if float(r["dice_all"]) >= dice_floor]
    best_hd_near = min(near_best_dice, key=lambda r: r["hd"])
    best_asd_near = min(near_best_dice, key=lambda r: r["asd"])
    top_k = max(1, int(args.top_k))
    summary = {
        "metadata": metadata,
        "num_candidates": len(rows),
        "pareto_dice_drop": float(args.pareto_dice_drop),
        "best_balanced": rows_by_score[0],
        "best_dice_all": best_dice,
        "best_hd_near_best_dice": best_hd_near,
        "best_asd_near_best_dice": best_asd_near,
        "top_by_score": rows_by_score[:top_k],
        "top_by_dice": sorted(rows, key=lambda r: r["dice_all"], reverse=True)[:top_k],
        "top_by_hd_near_best_dice": sorted(near_best_dice, key=lambda r: r["hd"])[:top_k],
        "top_by_asd_near_best_dice": sorted(near_best_dice, key=lambda r: r["asd"])[:top_k],
    }
    write_json(args.output_dir / "summary.json", summary)


def main() -> int:
    args = parse_args()
    if len(args.ckpts) != len(args.names):
        raise ValueError(f"--ckpts count ({len(args.ckpts)}) must match --names count ({len(args.names)}).")
    seed_everything(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cache = build_cache(args, device)

    probs_list = [arr.astype(np.float32) for arr in cache["probs"]]
    labels = cache["labels"].astype(bool)
    metadata = cache["metadata"]
    video_ids = metadata["video_ids"]
    label_cache = build_label_cache(labels)
    names = [str(n) for n in args.names]

    weight_rows = iter_weights(args)
    configs = iter_grid_configs(args)
    print(f"[grid] weights={len(weight_rows)} configs={len(configs)} candidates={len(weight_rows) * len(configs)}")

    fast_rows = []
    for weight_idx, weights in enumerate(weight_rows, start=1):
        probs = np.zeros_like(probs_list[0], dtype=np.float32)
        for weight, model_probs in zip(weights, probs_list):
            probs += float(weight) * model_probs
        for cfg in configs:
            _, metrics = evaluate_candidate(probs, label_cache, video_ids, cfg, args, include_distance=False)
            row = candidate_row(weights, cfg, metrics, names)
            row["fast_score"] = fast_score(row)
            fast_rows.append(row)
        if weight_idx % 25 == 0 or weight_idx == len(weight_rows):
            best_fast = max(fast_rows, key=fast_score)
            print(f"[fast] {weight_idx}/{len(weight_rows)} best={fast_score(best_fast):.6f} {best_fast}")

    write_csv(args.output_dir / "grid_fast_results.csv", sorted(fast_rows, key=fast_score, reverse=True))
    selected = select_full_metric_candidates(fast_rows, args)
    print(f"[full] selected={len(selected)} from {len(fast_rows)}")

    rows = []
    weight_keys = [f"weight_{name}" for name in names]
    for idx, fast_row in enumerate(selected, start=1):
        weights = tuple(float(fast_row[k]) for k in weight_keys)
        cfg = GridConfig(
            threshold=float(fast_row["threshold"]),
            min_area=int(fast_row["min_area"]),
            keep_top=int(fast_row["keep_top"]),
            fill_holes=bool(int(fast_row["fill_holes"])),
            open_iters=int(fast_row["open_iters"]),
            close_iters=int(fast_row["close_iters"]),
        )
        probs = np.zeros_like(probs_list[0], dtype=np.float32)
        for weight, model_probs in zip(weights, probs_list):
            probs += float(weight) * model_probs
        _, metrics = evaluate_candidate(probs, label_cache, video_ids, cfg, args, include_distance=True)
        row = candidate_row(weights, cfg, metrics, names)
        row["fast_score"] = float(fast_row["fast_score"])
        rows.append(row)
        if idx % 25 == 0 or idx == len(selected):
            best_full = max(rows, key=lambda r: r["score"])
            print(f"[full] {idx}/{len(selected)} best={best_full['score']:.6f} {best_full}")

    rows_by_score = sorted(rows, key=lambda r: r["score"], reverse=True)
    write_csv(args.output_dir / "grid_results.csv", rows_by_score)
    write_summary(args, rows, metadata)
    print(json.dumps({"best": rows_by_score[0], "summary": str(args.output_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
