#!/usr/bin/env python3
"""Evaluate Task2 checkpoint ensembles on the local validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete

from dataset import get_dataloaders
from model_factory import get_model
from utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task2 ensemble weights on val split.")
    parser.add_argument("--ckpt-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--names", type=str, nargs="+", required=True)
    parser.add_argument("--data-dir", type=str, default="data/reference_data/t2_tee/train")
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        required=True,
        help="Weight specs like 0.5,0.5 or 0.33,0.33,0.34. One spec per candidate.",
    )
    return parser.parse_args()


def load_ckpt_config(ckpt_path: Path) -> tuple[dict, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt, ckpt.get("args", {})


def build_model_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt, train_args = load_ckpt_config(ckpt_path)
    model = get_model(
        name=str(train_args.get("model", "unet3d")),
        model_size=str(train_args.get("model_size", "large")),
        in_channels=1,
        out_channels=int(train_args.get("num_classes", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, train_args


def parse_weight_specs(specs: list[str], n: int) -> list[list[float]]:
    parsed = []
    for spec in specs:
        weights = [float(x) for x in spec.split(",")]
        if len(weights) != n:
            raise ValueError(f"Weight spec {spec!r} has {len(weights)} values, expected {n}.")
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Weight spec {spec!r} must sum to a positive value.")
        parsed.append([w / total for w in weights])
    return parsed


def assert_compatible(configs: list[dict]) -> None:
    first = configs[0]
    keys = ("num_classes", "roi_size", "sw_batch_size", "enable_spacing_resample", "target_spacing")
    for cfg in configs[1:]:
        for key in keys:
            if cfg.get(key) != first.get(key):
                raise ValueError(f"Checkpoint configs differ for {key}: {first.get(key)} vs {cfg.get(key)}")


def new_metrics():
    return {
        "dice": DiceMetric(include_background=False, reduction="mean"),
        "hd": HausdorffDistanceMetric(
            include_background=False,
            distance_metric="euclidean",
            percentile=None,
            reduction="mean",
        ),
        "asd": SurfaceDistanceMetric(
            include_background=False,
            symmetric=True,
            reduction="mean",
        ),
    }


@torch.no_grad()
def evaluate_candidates(models, weights_list, loader, inferer, num_classes, device):
    metrics_by_candidate = [new_metrics() for _ in weights_list]
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    total = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        model_probs = []
        for model in models:
            logits = inferer(images, model)
            model_probs.append(torch.softmax(logits, dim=1))

        label_list = [post_label(x) for x in decollate_batch(labels)]
        for weights, metric_set in zip(weights_list, metrics_by_candidate):
            probs_sum = None
            for probs, weight in zip(model_probs, weights):
                weighted = probs * float(weight)
                probs_sum = weighted if probs_sum is None else probs_sum + weighted
            pred_list = [post_pred(x) for x in decollate_batch(probs_sum)]
            metric_set["dice"](y_pred=pred_list, y=label_list)
            metric_set["hd"](y_pred=pred_list, y=label_list)
            metric_set["asd"](y_pred=pred_list, y=label_list)
        print(f"[eval] {batch_idx}/{total}")

    results = []
    for metric_set in metrics_by_candidate:
        results.append(
            {
                "dsc": float(metric_set["dice"].aggregate().item()),
                "hd": float(metric_set["hd"].aggregate().item()),
                "asd": float(metric_set["asd"].aggregate().item()),
            }
        )
        metric_set["dice"].reset()
        metric_set["hd"].reset()
        metric_set["asd"].reset()
    return results


def main() -> int:
    args = parse_args()
    if len(args.names) != len(args.ckpt_paths):
        raise ValueError("--names length must match --ckpt-paths length.")

    device = get_device()
    models = []
    configs = []
    for ckpt_path in args.ckpt_paths:
        model, train_args = build_model_from_ckpt(ckpt_path, device)
        models.append(model)
        configs.append(train_args)
    assert_compatible(configs)

    cfg = configs[0]
    weights_list = parse_weight_specs(args.weights, len(args.ckpt_paths))
    roi_size = cfg.get("roi_size", [128, 128, 128])
    sw_batch_size = int(cfg.get("sw_batch_size", 1))
    num_classes = int(cfg.get("num_classes", 3))

    _, val_loader, _, val_files = get_dataloaders(
        data_dir=args.data_dir,
        val_count=args.val_count,
        roi_size=roi_size,
        batch_size=1,
        num_samples=1,
        num_workers=args.num_workers,
        num_classes=num_classes,
        enable_spacing_resample=bool(cfg.get("enable_spacing_resample", False)),
        target_spacing=cfg.get("target_spacing", [0.5, 0.5, 0.5]),
    )
    inferer = SlidingWindowInferer(
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        overlap=0.25,
        mode="gaussian",
    )

    print(f"Device: {device}")
    print(f"Validation cases: {len(val_files)}")
    print(f"Models: {args.names}")
    metrics_list = evaluate_candidates(models, weights_list, val_loader, inferer, num_classes, device)

    rows = []
    for weights, metrics in zip(weights_list, metrics_list):
        row = {
            "weights": weights,
            "weight_by_name": dict(zip(args.names, weights)),
            **metrics,
        }
        rows.append(row)
        print(
            "weights="
            + ",".join(f"{w:.3f}" for w in weights)
            + f" DSC={metrics['dsc']:.6f} HD={metrics['hd']:.4f} ASD={metrics['asd']:.4f}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt_paths": [str(p) for p in args.ckpt_paths],
                "names": args.names,
                "val_cases": [x["case_id"] for x in val_files],
                "results": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
