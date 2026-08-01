#!/usr/bin/env python3
"""Evaluate simple Task2 mask post-processing on the local validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete
from scipy import ndimage as ndi

from dataset import get_dataloaders
from model_factory import get_model
from utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task2 connected-component post-processing.")
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, default="data/reference_data/t2_tee/train")
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--keep-components", type=int, default=1)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--close-iters", type=int, default=0)
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


def postprocess_multiclass(
    mask: np.ndarray,
    num_classes: int,
    min_size: int,
    keep_components: int,
    fill_holes: bool,
    close_iters: int,
) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    structure = ndi.generate_binary_structure(mask.ndim, 1)

    for cls in range(1, num_classes):
        binary = mask == cls
        if not binary.any():
            continue

        if close_iters > 0:
            binary = ndi.binary_closing(binary, structure=structure, iterations=close_iters)
        if fill_holes:
            binary = ndi.binary_fill_holes(binary)

        labeled, n_labels = ndi.label(binary, structure=structure)
        if n_labels == 0:
            continue

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        component_ids = np.argsort(sizes)[::-1]
        kept = np.zeros_like(binary, dtype=bool)
        kept_count = 0
        for component_id in component_ids:
            if component_id == 0 or sizes[component_id] <= 0:
                continue
            if sizes[component_id] < min_size:
                continue
            kept |= labeled == component_id
            kept_count += 1
            if keep_components > 0 and kept_count >= keep_components:
                break

        out[kept] = cls

    return out


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


def aggregate_metrics(metric_set: dict) -> dict:
    result = {
        "dsc": float(metric_set["dice"].aggregate().item()),
        "hd": float(metric_set["hd"].aggregate().item()),
        "asd": float(metric_set["asd"].aggregate().item()),
    }
    for metric in metric_set.values():
        metric.reset()
    return result


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    model, train_args = build_model_from_ckpt(args.ckpt_path, device)

    num_classes = int(train_args.get("num_classes", 3))
    roi_size = train_args.get("roi_size", [128, 128, 128])
    sw_batch_size = int(train_args.get("sw_batch_size", 1))

    _, val_loader, _, val_files = get_dataloaders(
        data_dir=args.data_dir,
        val_count=args.val_count,
        roi_size=roi_size,
        batch_size=1,
        num_samples=1,
        num_workers=args.num_workers,
        num_classes=num_classes,
        enable_spacing_resample=bool(train_args.get("enable_spacing_resample", False)),
        target_spacing=train_args.get("target_spacing", [0.5, 0.5, 0.5]),
    )
    inferer = SlidingWindowInferer(
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        overlap=0.25,
        mode="gaussian",
    )

    raw_metrics = new_metrics()
    post_metrics = new_metrics()
    post_label = AsDiscrete(to_onehot=num_classes)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Validation cases: {len(val_files)}")
    print(
        "Postprocess: "
        f"min_size={args.min_size}, keep_components={args.keep_components}, "
        f"fill_holes={args.fill_holes}, close_iters={args.close_iters}"
    )

    case_rows = []
    total = len(val_loader)
    for batch_idx, batch in enumerate(val_loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = inferer(images, model)
        raw_mask = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        post_mask = postprocess_multiclass(
            raw_mask,
            num_classes=num_classes,
            min_size=args.min_size,
            keep_components=args.keep_components,
            fill_holes=args.fill_holes,
            close_iters=args.close_iters,
        )

        raw_pred = torch.as_tensor(raw_mask, device=device)[None, None]
        post_pred = torch.as_tensor(post_mask, device=device)[None, None]
        label_list = [post_label(x) for x in decollate_batch(labels)]
        raw_list = [post_label(x) for x in decollate_batch(raw_pred)]
        post_list = [post_label(x) for x in decollate_batch(post_pred)]

        raw_metrics["dice"](y_pred=raw_list, y=label_list)
        raw_metrics["hd"](y_pred=raw_list, y=label_list)
        raw_metrics["asd"](y_pred=raw_list, y=label_list)
        post_metrics["dice"](y_pred=post_list, y=label_list)
        post_metrics["hd"](y_pred=post_list, y=label_list)
        post_metrics["asd"](y_pred=post_list, y=label_list)

        raw_voxels = int(np.count_nonzero(raw_mask))
        post_voxels = int(np.count_nonzero(post_mask))
        case_id = str(batch["case_id"][0])
        case_rows.append(
            {
                "case_id": case_id,
                "raw_foreground_voxels": raw_voxels,
                "post_foreground_voxels": post_voxels,
                "removed_foreground_voxels": raw_voxels - post_voxels,
            }
        )
        print(f"[eval] {batch_idx}/{total} {case_id}: raw_fg={raw_voxels} post_fg={post_voxels}")

    raw_result = aggregate_metrics(raw_metrics)
    post_result = aggregate_metrics(post_metrics)
    delta = {
        "dsc": post_result["dsc"] - raw_result["dsc"],
        "hd": post_result["hd"] - raw_result["hd"],
        "asd": post_result["asd"] - raw_result["asd"],
    }

    result = {
        "ckpt_path": str(args.ckpt_path),
        "val_cases": [x["case_id"] for x in val_files],
        "postprocess": {
            "min_size": args.min_size,
            "keep_components": args.keep_components,
            "fill_holes": args.fill_holes,
            "close_iters": args.close_iters,
        },
        "raw": raw_result,
        "post": post_result,
        "delta_post_minus_raw": delta,
        "case_rows": case_rows,
    }

    print(
        "raw  "
        f"DSC={raw_result['dsc']:.6f} HD={raw_result['hd']:.4f} ASD={raw_result['asd']:.4f}"
    )
    print(
        "post "
        f"DSC={post_result['dsc']:.6f} HD={post_result['hd']:.4f} ASD={post_result['asd']:.4f}"
    )
    print(
        "delta "
        f"DSC={delta['dsc']:+.6f} HD={delta['hd']:+.4f} ASD={delta['asd']:+.4f}"
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
