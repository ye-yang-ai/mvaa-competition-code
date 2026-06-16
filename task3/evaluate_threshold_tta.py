#!/usr/bin/env python3
"""Fine-grained threshold and TTA evaluation for Task3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from dataset import (  # noqa: E402
    LabeledDataset,
    discover_samples,
    sample_has_foreground,
    split_train_val_by_video,
)
from model_factory import get_model  # noqa: E402
from train import dice_from_preds, predict_probs  # noqa: E402
from utils import MetricRefs, metric_quality_weighted, seed_everything  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "outputs" / "exp" / "task3_unetpp_res34_imagenet_e100_bs2" / "checkpoints" / "best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "task3_res34_imagenet_threshold_tta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task3 checkpoint across thresholds and TTA modes.")
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--labeled-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.50)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses checkpoint batch_size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta-modes", nargs="+", choices=["none", "tta"], default=["none", "tta"])
    return parser.parse_args()


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    return ckpt, train_args


def pick_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def threshold_values(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold-step must be positive")
    vals: List[float] = []
    cur = float(start)
    while cur <= float(stop) + step * 0.5:
        vals.append(round(cur, 6))
        cur += step
    if not vals:
        raise ValueError("No thresholds generated")
    return vals


def build_val_loader(train_args: dict, labeled_root_arg: Path | None, batch_size_arg: int, num_workers: int) -> tuple[DataLoader, dict]:
    seed = int(train_args.get("seed", 42))
    target_label = int(train_args.get("target_label", 10))
    val_video_count = int(train_args.get("val_video_count", 1))
    val_only_fg = bool(train_args.get("val_only_fg", True))
    max_val_samples = int(train_args.get("max_val_samples", 0))
    pseudo_train_only = bool(train_args.get("pseudo_train_only", True))
    cache_masks = bool(train_args.get("cache_masks", True))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    image_size = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    batch_size = int(batch_size_arg or train_args.get("batch_size", 2) or 2)

    labeled_root = labeled_root_arg or Path(str(train_args.get("labeled_root", REPO_ROOT / "data" / "t3_vid" / "train")))
    if not labeled_root.is_absolute():
        labeled_root = REPO_ROOT / labeled_root
    all_labeled = discover_samples(labeled_root)
    split_labeled = all_labeled
    if pseudo_train_only:
        split_labeled = [s for s in all_labeled if "_medsam2_pseudo" not in str(s.image_path)]

    _, val_samples, train_video_ids, val_video_ids = split_train_val_by_video(
        split_labeled,
        val_video_count=val_video_count,
        seed=seed,
    )
    val_samples_all = list(val_samples)
    if max_val_samples > 0:
        val_samples = val_samples[:max_val_samples]
        val_samples_all = val_samples_all[:max_val_samples]
    fg_before = len(val_samples)
    if val_only_fg:
        val_samples = [s for s in val_samples if sample_has_foreground(s, target_label=target_label)]
    fg_after = len(val_samples)
    if not val_samples:
        raise RuntimeError("Validation set is empty after foreground filtering.")

    ds = LabeledDataset(
        samples=val_samples,
        image_size=image_size,
        target_label=target_label,
        train=False,
        cache_masks=cache_masks,
        use_imagenet_norm=use_imagenet_norm,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
    )
    info = {
        "labeled_root": str(labeled_root),
        "all_labeled": len(all_labeled),
        "val_samples_all": len(val_samples_all),
        "val_samples": len(val_samples),
        "val_fg_before": fg_before,
        "val_fg_after": fg_after,
        "train_video_ids": train_video_ids,
        "val_video_ids": val_video_ids,
        "image_size": list(image_size),
        "batch_size": batch_size,
    }
    return loader, info


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


@torch.no_grad()
def collect_probs(model, loader: DataLoader, device: torch.device, use_amp: bool, use_tta: bool) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs_list = []
    labels_list = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        probs = predict_probs(model, images, use_amp=use_amp, use_tta=use_tta)
        probs_list.append(probs.float().cpu())
        labels_list.append(labels.float().cpu())
    return torch.cat(probs_list, dim=0), torch.cat(labels_list, dim=0)


def nanmean_to_float(x: torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(torch.nanmean(x).item())
    return float(x)


def evaluate_thresholds(
    probs: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Sequence[float],
    score_weights: dict,
) -> list[dict]:
    rows: list[dict] = []
    for thr in thresholds:
        preds = (probs > float(thr)).float()
        pred_pos_ratio = float(preds.mean().item())
        gt_pos_ratio = float(labels.mean().item())
        pred_non_empty = preds.flatten(1).sum(dim=1) > 0
        gt_non_empty = labels.flatten(1).sum(dim=1) > 0
        valid = pred_non_empty & gt_non_empty
        valid_cases = int(valid.sum().item())

        if valid_cases > 0:
            hd_metric = HausdorffDistanceMetric(
                include_background=True,
                distance_metric="euclidean",
                percentile=None,
                reduction="mean_batch",
            )
            asd_metric = SurfaceDistanceMetric(
                include_background=True,
                symmetric=True,
                reduction="mean_batch",
            )
            for i in range(0, valid_cases, 8):
                p = preds[valid][i : i + 8]
                y = labels[valid][i : i + 8]
                hd_metric(y_pred=p, y=y)
                asd_metric(y_pred=p, y=y)
            hd = nanmean_to_float(hd_metric.aggregate())
            asd = nanmean_to_float(asd_metric.aggregate())
            hd_metric.reset()
            asd_metric.reset()
        else:
            hd = float("nan")
            asd = float("nan")

        dice = float(dice_from_preds(preds, labels, ignore_empty_gt=True))
        score_obj = metric_quality_weighted(
            dsc=dice,
            hd=hd,
            asd=asd,
            refs=MetricRefs(hd_ref=float(score_weights["hd_ref"]), asd_ref=float(score_weights["asd_ref"])),
            dsc_weight=float(score_weights["dsc_weight"]),
            hd_weight=float(score_weights["hd_weight"]),
            asd_weight=float(score_weights["asd_weight"]),
        )
        rows.append(
            {
                "threshold": float(thr),
                "score": float(score_obj["score"]),
                "dice": dice,
                "hd": hd,
                "asd": asd,
                "q_dsc": float(score_obj["q_dsc"]),
                "q_hd": float(score_obj["q_hd"]),
                "q_asd": float(score_obj["q_asd"]),
                "pred_pos_ratio": pred_pos_ratio,
                "gt_pos_ratio": gt_pos_ratio,
                "valid_dist_cases": valid_cases,
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    ckpt, train_args = load_checkpoint(args.ckpt_path)
    seed_everything(int(train_args.get("seed", 42)))

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    val_loader, val_info = build_val_loader(train_args, args.labeled_root, args.batch_size, args.num_workers)

    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=None,
        encoder_weights_path=None,
        in_channels=3,
        classes=1,
    ).to(device)
    load_state_dict(model, ckpt)

    thresholds = threshold_values(args.threshold_start, args.threshold_stop, args.threshold_step)
    score_weights = {
        "dsc_weight": float(train_args.get("score_dsc_weight", 0.6)),
        "hd_weight": float(train_args.get("score_hd_weight", 0.2)),
        "asd_weight": float(train_args.get("score_asd_weight", 0.2)),
        "hd_ref": float(train_args.get("score_hd_ref", 20.0)),
        "asd_ref": float(train_args.get("score_asd_ref", 3.0)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}
    summary: dict = {
        "ckpt_path": str(args.ckpt_path),
        "device": str(device),
        "use_amp": use_amp,
        "arch": arch,
        "encoder_name": encoder_name,
        "val_info": val_info,
        "thresholds": thresholds,
        "score_weights": score_weights,
        "checkpoint_val_metrics": ckpt.get("val_metrics", {}),
        "modes": {},
    }

    for mode in args.tta_modes:
        use_tta = mode == "tta"
        print(f"Collecting probabilities: mode={mode} use_tta={use_tta}", flush=True)
        probs, labels = collect_probs(model, val_loader, device=device, use_amp=use_amp, use_tta=use_tta)
        rows = evaluate_thresholds(probs, labels, thresholds, score_weights)
        rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
        best = rows_sorted[0]
        all_results[mode] = rows
        write_csv(args.output_dir / f"threshold_sweep_{mode}.csv", rows)
        summary["modes"][mode] = {
            "use_tta": use_tta,
            "best": best,
            "top5": rows_sorted[:5],
        }
        print(
            "mode={mode} best_thr={thr:.3f} score={score:.6f} dice={dice:.6f} hd={hd:.4f} asd={asd:.4f}".format(
                mode=mode,
                thr=best["threshold"],
                score=best["score"],
                dice=best["dice"],
                hd=best["hd"],
                asd=best["asd"],
            ),
            flush=True,
        )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
