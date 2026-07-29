#!/usr/bin/env python3
"""Evaluate a two-model probability ensemble for Task3 on the internal split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from dataset import LabeledDataset, discover_samples, sample_has_foreground, split_train_val_by_video  # noqa: E402
from model_factory import get_model  # noqa: E402
from train import dice_from_preds, predict_probs  # noqa: E402
from utils import MetricRefs, metric_quality_weighted  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task3 probability ensemble.")
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--weights-a", type=float, nargs="+", default=[0.75, 0.65, 0.5])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument("--val-video-count", type=int, default=1)
    parser.add_argument("--val-only-fg", action="store_true", default=False)
    parser.add_argument("--no-val-only-fg", action="store_false", dest="val_only_fg")
    parser.add_argument("--score-dice-mode", choices=["fg", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
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
    args = ckpt.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return ckpt, args


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


def build_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt, train_args = load_ckpt(ckpt_path)
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    lemonfm_ckpt = train_args.get("lemonfm_ckpt", None)
    lemonfm_decoder_channels = int(train_args.get("lemonfm_decoder_channels", 128))
    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        lemonfm_ckpt=lemonfm_ckpt,
        lemonfm_decoder_channels=lemonfm_decoder_channels,
    ).to(device)
    load_state_dict(model, ckpt)
    model.eval()
    return model, train_args


def _nanmean_to_float(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.nanmean(x).item())


def distance_metrics(preds: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, int]:
    pred_non_empty = preds.flatten(1).sum(dim=1) > 0
    gt_non_empty = labels.flatten(1).sum(dim=1) > 0
    valid = pred_non_empty & gt_non_empty
    valid_cases = int(valid.sum().item())
    if valid_cases <= 0:
        return float("nan"), float("nan"), 0

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
    dist_preds = preds[valid]
    dist_labels = labels[valid]
    for i in range(0, dist_preds.shape[0], 8):
        hd_metric(y_pred=dist_preds[i : i + 8], y=dist_labels[i : i + 8])
        asd_metric(y_pred=dist_preds[i : i + 8], y=dist_labels[i : i + 8])
    hd = _nanmean_to_float(hd_metric.aggregate())
    asd = _nanmean_to_float(asd_metric.aggregate())
    hd_metric.reset()
    asd_metric.reset()
    return hd, asd, valid_cases


@torch.no_grad()
def collect_probs(
    model_a,
    args_a: dict,
    model_b,
    args_b: dict,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    use_tta: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size_a = tuple(int(v) for v in args_a.get("image_size", [448, 800]))
    size_b = tuple(int(v) for v in args_b.get("image_size", [448, 800]))
    probs_a = []
    probs_b = []
    labels = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        h, w = tuple(int(v) for v in y.shape[-2:])

        x_a = F.interpolate(images, size=size_a, mode="bilinear", align_corners=False)
        x_b = F.interpolate(images, size=size_b, mode="bilinear", align_corners=False)
        pa = predict_probs(model_a, x_a, use_amp=use_amp, use_tta=use_tta)
        pb = predict_probs(model_b, x_b, use_amp=use_amp, use_tta=use_tta)
        pa = F.interpolate(pa.float(), size=(h, w), mode="bilinear", align_corners=False)
        pb = F.interpolate(pb.float(), size=(h, w), mode="bilinear", align_corners=False)
        probs_a.append(pa.cpu())
        probs_b.append(pb.cpu())
        labels.append(y.float().cpu())
    return torch.cat(probs_a, dim=0), torch.cat(probs_b, dim=0), torch.cat(labels, dim=0)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    model_a, train_args_a = build_model(args.ckpt_a, device=device)
    model_b, train_args_b = build_model(args.ckpt_b, device=device)

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
        image_size=tuple(int(v) for v in train_args_a.get("image_size", [448, 800])),
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
    )

    pa, pb, labels = collect_probs(
        model_a=model_a,
        args_a=train_args_a,
        model_b=model_b,
        args_b=train_args_b,
        loader=loader,
        device=device,
        use_amp=use_amp,
        use_tta=bool(args.tta),
    )

    refs = MetricRefs(hd_ref=20.0, asd_ref=3.0)
    rows = []
    for wa in args.weights_a:
        wa = float(wa)
        wb = 1.0 - wa
        probs = wa * pa + wb * pb
        for thr in args.thresholds:
            preds = (probs > float(thr)).float()
            dsc_fg = dice_from_preds(preds, labels, ignore_empty_gt=True)
            dsc_all = dice_from_preds(preds, labels, ignore_empty_gt=False)
            dsc_for_score = dsc_all if str(args.score_dice_mode) == "all" else dsc_fg
            hd, asd, valid_cases = distance_metrics(preds, labels)
            pred_non_empty = preds.flatten(1).sum(dim=1) > 0
            gt_non_empty = labels.flatten(1).sum(dim=1) > 0
            empty_gt = ~gt_non_empty
            fg_gt = gt_non_empty
            empty_gt_count = int(empty_gt.sum().item())
            fg_gt_count = int(fg_gt.sum().item())
            empty_fp_count = int((empty_gt & pred_non_empty).sum().item())
            fg_miss_count = int((fg_gt & ~pred_non_empty).sum().item())
            empty_fp_rate = float(empty_fp_count / max(1, empty_gt_count))
            fg_miss_rate = float(fg_miss_count / max(1, fg_gt_count))
            presence_acc = float((pred_non_empty == gt_non_empty).float().mean().item())
            score = metric_quality_weighted(
                dsc=dsc_for_score,
                hd=hd,
                asd=asd,
                refs=refs,
                dsc_weight=0.6,
                hd_weight=0.2,
                asd_weight=0.2,
            )["score"]
            rows.append(
                {
                    "weight_a": wa,
                    "weight_b": wb,
                    "threshold": float(thr),
                    "dsc_fg": float(dsc_fg),
                    "dsc_all": float(dsc_all),
                    "hd": float(hd),
                    "asd": float(asd),
                    "score": float(score),
                    "valid_dist_cases": valid_cases,
                    "empty_fp_rate": empty_fp_rate,
                    "empty_fp_count": empty_fp_count,
                    "empty_gt_count": empty_gt_count,
                    "fg_miss_rate": fg_miss_rate,
                    "fg_miss_count": fg_miss_count,
                    "fg_gt_count": fg_gt_count,
                    "presence_acc": presence_acc,
                    "pred_pos_ratio": float(preds.mean().item()),
                    "gt_pos_ratio": float(labels.mean().item()),
                }
            )

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    result = {
        "ckpt_a": str(args.ckpt_a),
        "ckpt_b": str(args.ckpt_b),
        "val_videos": val_videos,
        "train_videos": train_videos,
        "num_val_samples": len(val_samples),
        "val_only_fg": bool(args.val_only_fg),
        "score_dice_mode": str(args.score_dice_mode),
        "device": str(device),
        "use_amp": use_amp,
        "use_tta": bool(args.tta),
        "best": rows_sorted[0],
        "rows": rows_sorted,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best": rows_sorted[0], "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
