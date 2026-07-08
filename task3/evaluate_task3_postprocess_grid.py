#!/usr/bin/env python3
"""Grid-search threshold and binary post-processing for Task3 validation masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from dataset import LabeledDataset, discover_samples, split_train_val_by_video  # noqa: E402
from model_factory import get_model  # noqa: E402
from train import predict_probs  # noqa: E402
from utils import MetricRefs, metric_quality_weighted, save_json, seed_everything  # noqa: E402


@dataclass(frozen=True)
class GridConfig:
    threshold: float
    min_area: int
    keep_top: int
    fill_holes: bool
    open_iters: int
    close_iters: int


@dataclass
class LabelCache:
    labels: np.ndarray
    flat_labels: np.ndarray
    gt_sum: np.ndarray
    gt_non_empty: np.ndarray
    label_surfaces: list[np.ndarray]
    dist_to_label_surfaces: list[np.ndarray | None]
    surface_valid: list[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task3 threshold/post-processing grid.")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=REPO_ROOT / "outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt",
    )
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/analysis/task3_postprocess_grid/v10")
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.50)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--min-areas", type=int, nargs="+", default=[0, 20, 50, 80, 120, 200, 500])
    parser.add_argument("--keep-tops", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--fill-holes", type=int, nargs="+", default=[0, 1], help="0/1 values")
    parser.add_argument("--open-iters", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--close-iters", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--dsc-weight", type=float, default=0.6)
    parser.add_argument("--hd-weight", type=float, default=0.2)
    parser.add_argument("--asd-weight", type=float, default=0.2)
    parser.add_argument("--hd-ref", type=float, default=20.0)
    parser.add_argument("--asd-ref", type=float, default=3.0)
    parser.add_argument("--balanced-dice-drop", type=float, default=0.015)
    parser.add_argument("--max-combos", type=int, default=0, help="Debug only; 0 means all")
    parser.add_argument("--fast-grid", action="store_true", default=True)
    parser.add_argument("--no-fast-grid", action="store_false", dest="fast_grid")
    parser.add_argument("--full-metric-top-k", type=int, default=80)
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    mode = str(device_arg).lower().strip()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(ckpt_path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    return ckpt, train_args


def load_state_dict(model: torch.nn.Module, ckpt_obj: Dict) -> None:
    if "model_state" in ckpt_obj:
        state = ckpt_obj["model_state"]
    elif "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    elif "state_dict" in ckpt_obj:
        state = ckpt_obj["state_dict"]
    else:
        state = ckpt_obj
    model.load_state_dict(state, strict=True)


def build_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    ckpt, train_args = load_ckpt(ckpt_path)
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)
    load_state_dict(model, ckpt)
    model.eval()
    return model, train_args, ckpt


def generated_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds is not None:
        return [float(x) for x in args.thresholds]
    n = int(round((float(args.threshold_stop) - float(args.threshold_start)) / float(args.threshold_step)))
    vals = [float(args.threshold_start) + i * float(args.threshold_step) for i in range(n + 1)]
    vals = [x for x in vals if x <= float(args.threshold_stop) + 1e-9]
    return [round(x, 6) for x in vals]


def iter_grid(args: argparse.Namespace) -> Iterable[GridConfig]:
    count = 0
    for threshold in generated_thresholds(args):
        for min_area in args.min_areas:
            for keep_top in args.keep_tops:
                for fill_holes in args.fill_holes:
                    for open_iters in args.open_iters:
                        for close_iters in args.close_iters:
                            yield GridConfig(
                                threshold=float(threshold),
                                min_area=int(min_area),
                                keep_top=int(keep_top),
                                fill_holes=bool(int(fill_holes)),
                                open_iters=int(open_iters),
                                close_iters=int(close_iters),
                            )
                            count += 1
                            if int(args.max_combos) > 0 and count >= int(args.max_combos):
                                return


@torch.no_grad()
def collect_validation_probs(
    model: torch.nn.Module,
    train_args: dict,
    samples,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    image_size = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    ds = LabeledDataset(
        samples=samples,
        image_size=image_size,
        target_label=int(args.target_label),
        train=False,
        cache_masks=True,
        use_imagenet_norm=use_imagenet_norm,
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

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float()
        h, w = tuple(int(v) for v in labels.shape[-2:])
        probs = predict_probs(model, images, use_amp=use_amp, use_tta=bool(args.tta))
        if tuple(probs.shape[-2:]) != (h, w):
            probs = F.interpolate(probs.float(), size=(h, w), mode="bilinear", align_corners=False)
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
        "image_size": image_size,
        "use_imagenet_norm": use_imagenet_norm,
    }


def surface_mask(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def surface_distances(pred: np.ndarray, target: np.ndarray) -> tuple[float, float] | None:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not bool(pred.any()) or not bool(target.any()):
        return None

    pred_surface = surface_mask(pred)
    target_surface = surface_mask(target)
    if not bool(pred_surface.any()) or not bool(target_surface.any()):
        return None

    dt_to_target = ndi.distance_transform_edt(~target_surface)
    dt_to_pred = ndi.distance_transform_edt(~pred_surface)
    d_pred_to_target = dt_to_target[pred_surface]
    d_target_to_pred = dt_to_pred[target_surface]
    if d_pred_to_target.size == 0 or d_target_to_pred.size == 0:
        return None
    hd = float(max(float(d_pred_to_target.max()), float(d_target_to_pred.max())))
    asd = float((float(d_pred_to_target.mean()) + float(d_target_to_pred.mean())) / 2.0)
    return hd, asd


def build_label_cache(labels: np.ndarray) -> LabelCache:
    labels = labels.astype(bool)
    flat_labels = labels.reshape(labels.shape[0], -1)
    gt_sum = flat_labels.sum(axis=1).astype(np.int64)
    gt_non_empty = gt_sum > 0
    label_surfaces: list[np.ndarray] = []
    dist_to_label_surfaces: list[np.ndarray | None] = []
    surface_valid: list[bool] = []
    for label in labels:
        if not bool(label.any()):
            label_surfaces.append(np.zeros_like(label, dtype=bool))
            dist_to_label_surfaces.append(None)
            surface_valid.append(False)
            continue
        surf = surface_mask(label)
        label_surfaces.append(surf)
        if bool(surf.any()):
            dist_to_label_surfaces.append(ndi.distance_transform_edt(~surf))
            surface_valid.append(True)
        else:
            dist_to_label_surfaces.append(None)
            surface_valid.append(False)
    return LabelCache(
        labels=labels,
        flat_labels=flat_labels,
        gt_sum=gt_sum,
        gt_non_empty=gt_non_empty,
        label_surfaces=label_surfaces,
        dist_to_label_surfaces=dist_to_label_surfaces,
        surface_valid=surface_valid,
    )


def cached_surface_distances(pred: np.ndarray, cache: LabelCache, idx: int) -> tuple[float, float] | None:
    pred = pred.astype(bool)
    if not bool(pred.any()) or not bool(cache.gt_non_empty[idx]) or not cache.surface_valid[idx]:
        return None
    pred_surface = surface_mask(pred)
    if not bool(pred_surface.any()):
        return None

    dt_to_label = cache.dist_to_label_surfaces[idx]
    if dt_to_label is None:
        return None
    label_surface = cache.label_surfaces[idx]
    dt_to_pred = ndi.distance_transform_edt(~pred_surface)
    d_pred_to_label = dt_to_label[pred_surface]
    d_label_to_pred = dt_to_pred[label_surface]
    if d_pred_to_label.size == 0 or d_label_to_pred.size == 0:
        return None
    hd = float(max(float(d_pred_to_label.max()), float(d_label_to_pred.max())))
    asd = float((float(d_pred_to_label.mean()) + float(d_label_to_pred.mean())) / 2.0)
    return hd, asd


def postprocess_binary(mask: np.ndarray, cfg: GridConfig) -> np.ndarray:
    out = mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    if cfg.open_iters > 0:
        out = ndi.binary_opening(out, structure=structure, iterations=int(cfg.open_iters))

    labeled, num_components = ndi.label(out)
    if num_components > 0:
        areas = np.bincount(labeled.ravel())
        keep = np.zeros(num_components + 1, dtype=bool)
        comp_ids = np.arange(1, num_components + 1)
        comp_areas = areas[1:]
        valid = comp_ids[comp_areas >= int(cfg.min_area)]
        if cfg.keep_top > 0 and valid.size > 0:
            valid_areas = areas[valid]
            order = np.argsort(valid_areas)[::-1][: int(cfg.keep_top)]
            valid = valid[order]
        keep[valid] = True
        out = keep[labeled]
    else:
        out = np.zeros_like(out, dtype=bool)

    if cfg.fill_holes:
        out = ndi.binary_fill_holes(out)
    if cfg.close_iters > 0:
        out = ndi.binary_closing(out, structure=structure, iterations=int(cfg.close_iters))
    return out.astype(bool)


def dice_from_counts(inter: np.ndarray, pred_sum: np.ndarray, gt_sum: np.ndarray, ignore_empty_gt: bool) -> float:
    denom = pred_sum + gt_sum
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    if ignore_empty_gt:
        valid = gt_sum > 0
        if not bool(valid.any()):
            return 0.0
        dice = dice[valid]
    return float(dice.mean())


def safe_mean(vals: Sequence[float]) -> float:
    clean = [float(v) for v in vals if not math.isnan(float(v))]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def compute_metrics(
    preds: np.ndarray,
    label_cache: LabelCache,
    video_ids: Sequence[str],
    include_per_video: bool = False,
    include_distance: bool = True,
) -> dict:
    labels = label_cache.labels
    preds_flat = preds.reshape(preds.shape[0], -1)
    pred_sum = preds_flat.sum(axis=1).astype(np.int64)
    inter = np.logical_and(preds_flat, label_cache.flat_labels).sum(axis=1).astype(np.int64)
    gt_sum = label_cache.gt_sum
    gt_non_empty = label_cache.gt_non_empty
    pred_non_empty = pred_sum > 0
    empty_mask = ~gt_non_empty
    fg_mask = gt_non_empty

    hds: list[float] = []
    asds: list[float] = []
    valid_dist_cases = 0
    if include_distance:
        for i, p in enumerate(preds):
            dist = cached_surface_distances(p, label_cache, i)
            if dist is None:
                continue
            hd, asd = dist
            hds.append(hd)
            asds.append(asd)
            valid_dist_cases += 1

    out = {
        "dice_all": dice_from_counts(inter, pred_sum, gt_sum, ignore_empty_gt=False),
        "dice_fg": dice_from_counts(inter, pred_sum, gt_sum, ignore_empty_gt=True),
        "hd": safe_mean(hds) if include_distance else float("nan"),
        "asd": safe_mean(asds) if include_distance else float("nan"),
        "valid_dist_cases": int(valid_dist_cases),
        "pred_pos_ratio": float(preds.mean()),
        "gt_pos_ratio": float(labels.mean()),
        "empty_frames": int(empty_mask.sum()),
        "empty_fp_rate": float(pred_non_empty[empty_mask].mean()) if bool(empty_mask.any()) else float("nan"),
        "fg_frames": int(fg_mask.sum()),
        "num_frames": int(labels.shape[0]),
    }
    if include_per_video:
        per_video = []
        for video_id in sorted(set(video_ids)):
            idx = np.asarray([v == video_id for v in video_ids], dtype=bool)
            v_preds = preds[idx]
            v_labels = labels[idx]
            v_pred_sum = pred_sum[idx]
            v_gt_sum = gt_sum[idx]
            v_inter = inter[idx]
            v_gt_non_empty = v_gt_sum > 0
            v_pred_non_empty = v_pred_sum > 0
            v_empty = ~v_gt_non_empty
            v_hds: list[float] = []
            v_asds: list[float] = []
            original_indices = np.nonzero(idx)[0]
            if include_distance:
                for original_idx, p in zip(original_indices, v_preds):
                    dist = cached_surface_distances(p, label_cache, int(original_idx))
                    if dist is None:
                        continue
                    hd, asd = dist
                    v_hds.append(hd)
                    v_asds.append(asd)
            per_video.append(
                {
                    "video_id": video_id,
                    "num_frames": int(idx.sum()),
                    "num_fg_frames": int(v_gt_non_empty.sum()),
                    "dice_all": dice_from_counts(v_inter, v_pred_sum, v_gt_sum, ignore_empty_gt=False),
                    "dice_fg": dice_from_counts(v_inter, v_pred_sum, v_gt_sum, ignore_empty_gt=True),
                    "hd": safe_mean(v_hds) if include_distance else float("nan"),
                    "asd": safe_mean(v_asds) if include_distance else float("nan"),
                    "empty_fp_rate": float(v_pred_non_empty[v_empty].mean()) if bool(v_empty.any()) else float("nan"),
                    "pred_pos_ratio": float(v_preds.mean()),
                    "gt_pos_ratio": float(v_labels.mean()),
                }
            )
        out["per_video"] = per_video
    return out


def evaluate_config(
    probs: np.ndarray,
    label_cache: LabelCache,
    video_ids: Sequence[str],
    cfg: GridConfig,
    include_per_video: bool = False,
    include_distance: bool = True,
) -> tuple[np.ndarray, dict]:
    raw = probs > float(cfg.threshold)
    preds = np.zeros_like(raw, dtype=bool)
    for i in range(raw.shape[0]):
        preds[i] = postprocess_binary(raw[i], cfg)
    return preds, compute_metrics(
        preds,
        label_cache,
        video_ids,
        include_per_video=include_per_video,
        include_distance=include_distance,
    )


def row_from_metrics(cfg: GridConfig, metrics: dict, score: float) -> dict:
    return {
        "threshold": float(cfg.threshold),
        "min_area": int(cfg.min_area),
        "keep_top": int(cfg.keep_top),
        "fill_holes": int(cfg.fill_holes),
        "open_iters": int(cfg.open_iters),
        "close_iters": int(cfg.close_iters),
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


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_row(row: dict) -> dict:
    out = dict(row)
    for key, value in list(out.items()):
        if isinstance(value, float):
            out[key] = None if math.isnan(value) else value
    return out


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    model, train_args, ckpt = build_model(args.ckpt_path, device=device)

    all_samples = discover_samples(args.labeled_root)
    _, val_samples, train_videos, val_videos = split_train_val_by_video(
        all_samples,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )
    print(
        f"[setup] device={device} amp={use_amp} tta={bool(args.tta)} "
        f"val_frames={len(val_samples)} val_videos={val_videos}"
    )

    collected = collect_validation_probs(
        model=model,
        train_args=train_args,
        samples=val_samples,
        args=args,
        device=device,
    )
    probs = collected["probs"]
    labels = collected["labels"]
    label_cache = build_label_cache(labels)
    video_ids = collected["video_ids"]

    baseline_thr = float(ckpt.get("val_metrics", {}).get("val_threshold", 0.25))
    baseline_cfg = GridConfig(
        threshold=baseline_thr,
        min_area=0,
        keep_top=0,
        fill_holes=False,
        open_iters=0,
        close_iters=0,
    )
    baseline_preds, baseline_metrics = evaluate_config(
        probs,
        label_cache,
        video_ids,
        baseline_cfg,
        include_per_video=True,
        include_distance=True,
    )
    del baseline_preds

    refs = MetricRefs(hd_ref=float(args.hd_ref), asd_ref=float(args.asd_ref))
    rows = []
    per_video_by_key = {}
    configs = list(iter_grid(args))
    total = len(configs)
    print(f"[grid] evaluating {total} configs")
    for idx, cfg in enumerate(configs, start=1):
        _, metrics = evaluate_config(
            probs,
            label_cache,
            video_ids,
            cfg,
            include_distance=not bool(args.fast_grid),
        )
        if bool(args.fast_grid):
            pos_gap = abs(float(metrics["pred_pos_ratio"]) - float(metrics["gt_pos_ratio"]))
            empty_fp = 0.0 if math.isnan(float(metrics["empty_fp_rate"])) else float(metrics["empty_fp_rate"])
            score = float(metrics["dice_fg"]) - 0.04 * empty_fp - 0.5 * pos_gap
        else:
            score = metric_quality_weighted(
                dsc=metrics["dice_fg"],
                hd=metrics["hd"],
                asd=metrics["asd"],
                refs=refs,
                dsc_weight=float(args.dsc_weight),
                hd_weight=float(args.hd_weight),
                asd_weight=float(args.asd_weight),
            )["score"]
        row = row_from_metrics(cfg, metrics, score)
        rows.append(row)
        key = (
            f"thr={cfg.threshold:.6f}|min={cfg.min_area}|keep={cfg.keep_top}|"
            f"fill={int(cfg.fill_holes)}|open={cfg.open_iters}|close={cfg.close_iters}"
        )
        if idx == 1 or idx == total or idx % 100 == 0:
            print(f"[grid] {idx}/{total} best_score={max(r['score'] for r in rows):.6f}")

    rows_by_score = sorted(rows, key=lambda r: r["score"], reverse=True)
    rows_by_dice = sorted(rows, key=lambda r: r["dice_fg"], reverse=True)
    rows_by_hd = sorted(rows, key=lambda r: (math.inf if math.isnan(r["hd"]) else r["hd"], -r["dice_fg"]))
    min_dice = float(baseline_metrics["dice_fg"]) - float(args.balanced_dice_drop)
    balanced_candidates = [r for r in rows if r["dice_fg"] >= min_dice]
    rows_balanced = sorted(
        balanced_candidates,
        key=lambda r: (
            math.inf if math.isnan(r["hd"]) else r["hd"],
            math.inf if math.isnan(r["asd"]) else r["asd"],
            -r["dice_fg"],
        ),
    )

    if bool(args.fast_grid):
        preselect = []
        seen_keys = set()
        for source in (
            rows_by_score[: int(args.full_metric_top_k)],
            rows_by_dice[: max(10, int(args.full_metric_top_k) // 4)],
            rows_balanced[: int(args.full_metric_top_k)],
        ):
            for row in source:
                key = (
                    float(row["threshold"]),
                    int(row["min_area"]),
                    int(row["keep_top"]),
                    int(row["fill_holes"]),
                    int(row["open_iters"]),
                    int(row["close_iters"]),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                preselect.append(row)
        print(f"[full] evaluating distance metrics for {len(preselect)} selected configs")
        full_rows_by_key = {}
        for idx, row in enumerate(preselect, start=1):
            cfg = GridConfig(
                threshold=float(row["threshold"]),
                min_area=int(row["min_area"]),
                keep_top=int(row["keep_top"]),
                fill_holes=bool(int(row["fill_holes"])),
                open_iters=int(row["open_iters"]),
                close_iters=int(row["close_iters"]),
            )
            _, metrics = evaluate_config(probs, label_cache, video_ids, cfg, include_distance=True)
            score = metric_quality_weighted(
                dsc=metrics["dice_fg"],
                hd=metrics["hd"],
                asd=metrics["asd"],
                refs=refs,
                dsc_weight=float(args.dsc_weight),
                hd_weight=float(args.hd_weight),
                asd_weight=float(args.asd_weight),
            )["score"]
            full_row = row_from_metrics(cfg, metrics, score)
            key = (
                float(full_row["threshold"]),
                int(full_row["min_area"]),
                int(full_row["keep_top"]),
                int(full_row["fill_holes"]),
                int(full_row["open_iters"]),
                int(full_row["close_iters"]),
            )
            full_rows_by_key[key] = full_row
            if idx == 1 or idx == len(preselect) or idx % 20 == 0:
                print(f"[full] {idx}/{len(preselect)}")
        merged = []
        for row in rows:
            key = (
                float(row["threshold"]),
                int(row["min_area"]),
                int(row["keep_top"]),
                int(row["fill_holes"]),
                int(row["open_iters"]),
                int(row["close_iters"]),
            )
            merged.append(full_rows_by_key.get(key, row))
        rows = merged
        full_rows = list(full_rows_by_key.values())
        rows_by_score = sorted(full_rows, key=lambda r: r["score"], reverse=True)
        rows_by_dice = sorted(full_rows, key=lambda r: r["dice_fg"], reverse=True)
        rows_by_hd = sorted(full_rows, key=lambda r: (math.inf if math.isnan(r["hd"]) else r["hd"], -r["dice_fg"]))
        balanced_candidates = [r for r in full_rows if r["dice_fg"] >= min_dice]
        rows_balanced = sorted(
            balanced_candidates,
            key=lambda r: (
                math.inf if math.isnan(r["hd"]) else r["hd"],
                math.inf if math.isnan(r["asd"]) else r["asd"],
                -r["dice_fg"],
            ),
        )

    top_rows_for_per_video = []
    seen_keys = set()
    for source in (rows_by_score[:10], rows_by_dice[:5], rows_by_hd[:5], rows_balanced[:10]):
        for row in source:
            key = (
                float(row["threshold"]),
                int(row["min_area"]),
                int(row["keep_top"]),
                int(row["fill_holes"]),
                int(row["open_iters"]),
                int(row["close_iters"]),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            top_rows_for_per_video.append(row)

    for row in top_rows_for_per_video:
        cfg = GridConfig(
            threshold=float(row["threshold"]),
            min_area=int(row["min_area"]),
            keep_top=int(row["keep_top"]),
            fill_holes=bool(int(row["fill_holes"])),
            open_iters=int(row["open_iters"]),
            close_iters=int(row["close_iters"]),
        )
        _, metrics = evaluate_config(probs, label_cache, video_ids, cfg, include_per_video=True)
        key = (
            f"thr={cfg.threshold:.6f}|min={cfg.min_area}|keep={cfg.keep_top}|"
            f"fill={int(cfg.fill_holes)}|open={cfg.open_iters}|close={cfg.close_iters}"
        )
        per_video_by_key[key] = metrics["per_video"]
    per_video_by_key["baseline"] = baseline_metrics["per_video"]

    write_csv(args.output_dir / "grid_results.csv", rows_by_score)
    with (args.output_dir / "grid_results.json").open("w", encoding="utf-8") as f:
        json.dump([compact_row(r) for r in rows_by_score], f, ensure_ascii=False, indent=2)
    with (args.output_dir / "per_video_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_video_by_key, f, ensure_ascii=False, indent=2)

    summary = {
        "ckpt_path": str(args.ckpt_path),
        "labeled_root": str(args.labeled_root),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "use_amp": use_amp,
        "use_tta": bool(args.tta),
        "train_args": train_args,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "num_val_samples": len(val_samples),
        "image_size": collected["image_size"],
        "baseline_cfg": baseline_cfg.__dict__,
        "baseline": compact_row(
            row_from_metrics(
                baseline_cfg,
                baseline_metrics,
                metric_quality_weighted(
                    dsc=baseline_metrics["dice_fg"],
                    hd=baseline_metrics["hd"],
                    asd=baseline_metrics["asd"],
                    refs=refs,
                    dsc_weight=float(args.dsc_weight),
                    hd_weight=float(args.hd_weight),
                    asd_weight=float(args.asd_weight),
                )["score"],
            )
        ),
        "best_by_score": compact_row(rows_by_score[0]),
        "best_by_dice": compact_row(rows_by_dice[0]),
        "best_by_hd": compact_row(rows_by_hd[0]),
        "balanced_min_dice": min_dice,
        "best_balanced": compact_row(rows_balanced[0]) if rows_balanced else None,
        "top10_by_score": [compact_row(r) for r in rows_by_score[:10]],
        "top10_balanced": [compact_row(r) for r in rows_balanced[:10]],
        "grid": {
            "thresholds": generated_thresholds(args),
            "min_areas": args.min_areas,
            "keep_tops": args.keep_tops,
            "fill_holes": args.fill_holes,
            "open_iters": args.open_iters,
            "close_iters": args.close_iters,
        },
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["baseline", "best_by_score", "best_by_dice", "best_by_hd", "best_balanced"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
