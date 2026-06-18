#!/usr/bin/env python3
"""Evaluate connected-component and morphology post-processing on Task3 val split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from scipy import ndimage

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import LabeledDataset, Sample  # noqa: E402
from medsam2_backbone import Task3MedSAM2EncoderSeg  # noqa: E402
from medsam2_backbone.build_encoder import DEFAULT_ENCODER_CFG, DEFAULT_ENCODER_CKPT  # noqa: E402
from utils import ensure_dir, save_json  # noqa: E402


@dataclass(frozen=True)
class PostprocessConfig:
    min_area: int
    keep_components: int
    close_iters: int
    fill_holes: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task3 MedSAM2 post-processing configs.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "outputs" / "medsam2_stageA" / "task3_frozen")
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument("--min-areas", type=int, nargs="+", default=[0, 25, 50, 100, 200, 400])
    parser.add_argument("--keep-components", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--close-iters", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fill-holes", action="store_true", default=True)
    parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes")
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_from_image_path(image_path: str, target_label: int) -> Sample:
    p = Path(image_path)
    stem = p.stem
    label_bin = p.with_name(f"{stem}_label_bin.png")
    label_tar = p.with_name(f"{stem}_png_Label.tar")
    if label_bin.exists():
        return Sample(p, label_bin, "bin_png", stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1]))
    if label_tar.exists():
        return Sample(p, label_tar, "tar", stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1]))
    raise FileNotFoundError(f"No label found for {p}")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    model = Task3MedSAM2EncoderSeg(
        encoder_cfg=str(train_args.get("encoder_cfg", DEFAULT_ENCODER_CFG)),
        encoder_ckpt=str(train_args.get("encoder_ckpt", DEFAULT_ENCODER_CKPT)),
        decoder_channels=int(train_args.get("decoder_channels", 128)),
        freeze_encoder=bool(train_args.get("freeze_encoder", True)),
        decoder_version=str(train_args.get("decoder_version", "v1")),
        device=device,
    ).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, train_args, ckpt


@torch.no_grad()
def collect_probs(model, loader, device: torch.device, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    probs_list = []
    labels_list = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        probs_list.append(torch.sigmoid(logits).float().cpu())
        labels_list.append(labels.cpu())
    return torch.cat(probs_list, dim=0), torch.cat(labels_list, dim=0)


def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_dilation(mask.astype(bool), structure=structure, iterations=max(0, int(iterations)))


def binary_erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_erosion(mask.astype(bool), structure=structure, iterations=max(0, int(iterations)))


def binary_close(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_closing(mask.astype(bool), structure=structure, iterations=int(iterations))


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, List[int]]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, num = ndimage.label(mask.astype(bool), structure=structure)
    if num == 0:
        return labels.astype(np.int32), []
    areas = ndimage.sum(mask.astype(np.uint8), labels=labels, index=np.arange(1, num + 1))
    return labels.astype(np.int32), [int(x) for x in np.asarray(areas).tolist()]


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask.astype(bool))


def apply_postprocess(mask: np.ndarray, cfg: PostprocessConfig) -> np.ndarray:
    out = mask.astype(bool)
    out = binary_close(out, cfg.close_iters)
    if cfg.fill_holes:
        out = fill_binary_holes(out)

    labels, areas = connected_components(out)
    if not areas:
        return out.astype(np.uint8)

    keep = set(range(1, len(areas) + 1))
    if cfg.min_area > 0:
        keep = {idx for idx in keep if areas[idx - 1] >= int(cfg.min_area)}
    if cfg.keep_components > 0:
        ranked = sorted(keep, key=lambda idx: areas[idx - 1], reverse=True)
        keep = set(ranked[: int(cfg.keep_components)])
    if not keep:
        return np.zeros_like(out, dtype=np.uint8)
    return np.isin(labels, list(keep)).astype(np.uint8)


def dice_from_preds(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (preds * labels).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))
    valid = labels.sum(dim=(1, 2, 3)) > 0
    if bool(valid.any()):
        dice = (2.0 * inter[valid] + eps) / (denom[valid] + eps)
        return float(dice.mean().item())
    return 0.0


def _nanmean_to_float(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.nanmean(x).item())


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    dice = dice_from_preds(preds, labels)
    pred_non_empty = preds.flatten(1).sum(dim=1) > 0
    gt_non_empty = labels.flatten(1).sum(dim=1) > 0
    valid = pred_non_empty & gt_non_empty
    valid_cases = int(valid.sum().item())
    if valid_cases > 0:
        hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=None, reduction="mean_batch")
        asd_metric = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="mean_batch")
        hd_metric(y_pred=preds[valid], y=labels[valid])
        asd_metric(y_pred=preds[valid], y=labels[valid])
        hd = _nanmean_to_float(hd_metric.aggregate())
        asd = _nanmean_to_float(asd_metric.aggregate())
        hd_metric.reset()
        asd_metric.reset()
    else:
        hd = float("nan")
        asd = float("nan")
    return {
        "dice": dice,
        "hd": hd,
        "asd": asd,
        "pred_pos_ratio": float(preds.mean().item()),
        "gt_pos_ratio": float(labels.mean().item()),
        "valid_dist_cases": valid_cases,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    ckpt_path = args.ckpt_path or (run_dir / "checkpoints" / "best.pt")
    output_dir = ensure_dir(args.output_dir or (run_dir / "postprocess_eval"))

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    model, train_args, ckpt = load_model(ckpt_path, device)

    split = load_json(run_dir / "split.json")
    target_label = int(train_args.get("target_label", 10))
    image_size = tuple(int(v) for v in train_args.get("image_size", [512, 512]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    val_paths = split["val_internal_samples"]
    samples = [sample_from_image_path(path, target_label=target_label) for path in val_paths]

    ds = LabeledDataset(
        samples=samples,
        image_size=image_size,
        target_label=target_label,
        train=False,
        cache_masks=True,
        use_imagenet_norm=use_imagenet_norm,
        seed=int(train_args.get("seed", 42)),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    print(f"Device: {device} AMP={use_amp}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Val samples: {len(samples)}")
    print(f"Checkpoint val threshold: {ckpt.get('val_metrics', {}).get('val_threshold', 'unknown')}")

    probs, labels = collect_probs(model, loader, device=device, use_amp=use_amp)
    labels = labels.float()

    configs: List[PostprocessConfig] = []
    for min_area in args.min_areas:
        for keep in args.keep_components:
            for close_iter in args.close_iters:
                configs.append(
                    PostprocessConfig(
                        min_area=int(min_area),
                        keep_components=int(keep),
                        close_iters=int(close_iter),
                        fill_holes=bool(args.fill_holes),
                    )
                )

    rows = []
    best = None
    for thr in args.thresholds:
        raw_preds = (probs > float(thr)).float()
        raw_metrics = compute_metrics(raw_preds, labels)
        rows.append(
            {
                "threshold": float(thr),
                "min_area": 0,
                "keep_components": 0,
                "close_iters": 0,
                "fill_holes": False,
                "kind": "raw",
                **raw_metrics,
            }
        )

        probs_np = probs[:, 0].numpy()
        for cfg in configs:
            processed = []
            for i in range(probs_np.shape[0]):
                mask = probs_np[i] > float(thr)
                processed.append(apply_postprocess(mask, cfg))
            pred_t = torch.from_numpy(np.stack(processed, axis=0)[:, None]).float()
            metrics = compute_metrics(pred_t, labels)
            row = {
                "threshold": float(thr),
                "min_area": int(cfg.min_area),
                "keep_components": int(cfg.keep_components),
                "close_iters": int(cfg.close_iters),
                "fill_holes": bool(cfg.fill_holes),
                "kind": "post",
                **metrics,
            }
            rows.append(row)
            if best is None or row["dice"] > best["dice"]:
                best = row

    csv_path = output_dir / "postprocess_grid.csv"
    fieldnames = [
        "kind",
        "threshold",
        "min_area",
        "keep_components",
        "close_iters",
        "fill_holes",
        "dice",
        "hd",
        "asd",
        "pred_pos_ratio",
        "gt_pos_ratio",
        "valid_dist_cases",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    raw_best = max((r for r in rows if r["kind"] == "raw"), key=lambda r: r["dice"])
    result = {
        "ckpt_path": str(ckpt_path),
        "samples": len(samples),
        "raw_best": raw_best,
        "post_best": best,
        "csv": str(csv_path),
    }
    save_json(output_dir / "postprocess_summary.json", result)

    def fmt_metric(row) -> str:
        hd_s = "nan" if math.isnan(row["hd"]) else f"{row['hd']:.2f}"
        asd_s = "nan" if math.isnan(row["asd"]) else f"{row['asd']:.2f}"
        return (
            f"thr={row['threshold']:.2f} dice={row['dice']:.4f} "
            f"hd={hd_s} "
            f"asd={asd_s} "
            f"pred_pos={row['pred_pos_ratio']:.4f}"
        )

    print(f"Raw best:  {fmt_metric(raw_best)}")
    if best is not None:
        print(
            "Post best: "
            f"{fmt_metric(best)} min_area={best['min_area']} "
            f"keep={best['keep_components']} close={best['close_iters']} fill={best['fill_holes']}"
        )
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
