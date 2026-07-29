#!/usr/bin/env python3
"""Train an independent Task3 frame-level presence/empty gate.

This gate is intentionally decoupled from the segmentation head. It predicts
whether a frame contains the target label, then validation can zero out
segmentation masks only for frames classified as empty.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import LabeledDataset, discover_samples, split_train_val_by_video
from evaluate_task3_postprocess_grid import build_label_cache, compute_metrics
from model_factory import DEFAULT_LEMONFM_CKPT, get_model
from train import predict_probs as predict_seg_probs
from utils import ensure_dir, get_device, save_json, seed_everything, setup_logger

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


class LemonFMPresenceClassifier(nn.Module):
    """LemonFM ConvNeXt-Large encoder with a binary frame-level head."""

    def __init__(
        self,
        pretrained_weights: str | Path = DEFAULT_LEMONFM_CKPT,
        dropout: float = 0.2,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        base = torchvision.models.convnext_large(weights=None)
        in_features = int(base.classifier[2].in_features)
        base.classifier[2] = nn.Identity()
        self._load_lemonfm_weights(base, Path(pretrained_weights))

        if int(hidden_dim) > 0:
            head: nn.Module = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(in_features, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, 1))
        base.classifier[2] = head
        self.net = base

    @staticmethod
    def _load_lemonfm_weights(base: nn.Module, weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"LemonFM checkpoint not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "teacher" not in ckpt:
            raise ValueError(f"LemonFM checkpoint must contain a 'teacher' state dict: {weights_path}")
        state_dict = {
            k.replace("backbone.", "", 1): v
            for k, v in ckpt["teacher"].items()
            if k.startswith("backbone.")
        }
        msg = base.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                "Failed to load LemonFM backbone exactly: "
                f"missing={msg.missing_keys[:10]} unexpected={msg.unexpected_keys[:10]}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()

    def encoder_parameters(self):
        return self.net.features.parameters()

    def head_parameters(self):
        return self.net.classifier.parameters()

    def set_encoder_tail_trainable(self, tail_modules: int) -> list[str]:
        n_tail = max(0, int(tail_modules))
        for p in self.encoder_parameters():
            p.requires_grad_(False)
        if n_tail <= 0:
            return []

        modules = list(self.net.features.children())
        start = max(0, len(modules) - n_tail)
        names: list[str] = []
        for idx, module in enumerate(modules[start:], start=start):
            for p in module.parameters():
                p.requires_grad_(True)
            names.append(f"features[{idx}]")
        return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task3 presence/empty classifier gate.")
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/opt/task3/t3_lemonfm_presence_gate_s51")
    parser.add_argument("--lemonfm-ckpt", type=Path, default=DEFAULT_LEMONFM_CKPT)
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=[448, 800], help="H W")
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder")
    parser.add_argument("--encoder-unfreeze-tail", type=int, default=0)

    parser.add_argument("--balance-presence-sampling", action="store_true", default=True)
    parser.add_argument("--no-balance-presence-sampling", action="store_false", dest="balance_presence_sampling")
    parser.add_argument("--presence-pos-loss-weight", type=float, default=2.0)
    parser.add_argument("--presence-empty-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-fg-miss", type=int, default=0)
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[round(float(x), 3) for x in np.linspace(0.01, 0.99, 99)],
    )

    parser.add_argument("--seg-ckpt-path", type=Path, default=None)
    parser.add_argument("--seg-threshold", type=float, default=-1.0)
    parser.add_argument("--seg-tta", action="store_true", default=True)
    parser.add_argument("--no-seg-tta", action="store_false", dest="seg_tta")

    parser.add_argument("--use-imagenet-norm", action="store_true", default=True)
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.add_argument("--cache-masks", action="store_true", default=True)
    parser.add_argument("--no-cache-masks", action="store_false", dest="cache_masks")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=10)
    return parser.parse_args()


def tensor_presence(labels: torch.Tensor) -> torch.Tensor:
    return (labels.flatten(1).sum(dim=1) > 0).float()


def build_presence_sampler(dataset: LabeledDataset) -> WeightedRandomSampler:
    targets = np.asarray([r > 0.0 for r in dataset.sample_fg_ratio], dtype=bool)
    n = int(targets.size)
    pos = max(1, int(targets.sum()))
    neg = max(1, int((~targets).sum()))
    weights = np.where(targets, n / (2.0 * pos), n / (2.0 * neg)).astype(np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=n, replacement=True)


def configure_trainable(model: LemonFMPresenceClassifier, args: argparse.Namespace, logger) -> None:
    if bool(args.freeze_encoder) or int(args.encoder_unfreeze_tail) > 0:
        for p in model.encoder_parameters():
            p.requires_grad_(False)
    names = []
    if int(args.encoder_unfreeze_tail) > 0:
        names = model.set_encoder_tail_trainable(int(args.encoder_unfreeze_tail))
        logger.info("Encoder selective unfreeze: tail=%d modules=%s", int(args.encoder_unfreeze_tail), names)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_total = sum(p.numel() for p in model.encoder_parameters())
    enc_trainable = sum(p.numel() for p in model.encoder_parameters() if p.requires_grad)
    logger.info(
        "Trainable params: total=%d trainable=%d encoder=%d trainable_encoder=%d",
        total,
        trainable,
        enc_total,
        enc_trainable,
    )


def build_optimizer(model: LemonFMPresenceClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder_params = [p for p in model.encoder_parameters() if p.requires_grad]
    head_params = [p for p in model.head_parameters() if p.requires_grad]
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": float(args.lr)})
    if encoder_params:
        enc_lr = float(args.encoder_lr) if float(args.encoder_lr) > 0 else float(args.lr) * 0.1
        groups.append({"params": encoder_params, "lr": enc_lr})
    if not groups:
        raise RuntimeError("No trainable parameters for presence gate.")
    return torch.optim.AdamW(groups, lr=float(args.lr), weight_decay=float(args.weight_decay))


def lr_factor(epoch: int, args: argparse.Namespace) -> float:
    warmup = max(0, int(args.warmup_epochs))
    if warmup > 0 and epoch <= warmup:
        return float(epoch) / float(warmup)
    total = max(1, int(args.epochs) - warmup)
    progress = min(1.0, max(0.0, float(epoch - warmup) / float(total)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(float(args.min_lr) / max(float(args.lr), 1e-12), cosine)


def set_lrs(optimizer: torch.optim.Optimizer, base_lrs: Sequence[float], factor: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = max(0.0, float(base_lr) * float(factor))


@torch.no_grad()
def predict_presence_probs(
    model: nn.Module,
    images: torch.Tensor,
    use_amp: bool,
    use_tta: bool,
) -> torch.Tensor:
    device_type = images.device.type
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        probs = torch.sigmoid(model(images))
    if not use_tta:
        return probs

    probs_sum = probs
    for dims in [(3,), (2,), (2, 3)]:
        x = torch.flip(images, dims=dims)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            probs_f = torch.sigmoid(model(x))
        probs_sum = probs_sum + probs_f
    return probs_sum / 4.0


def choose_gate_threshold(probs: np.ndarray, targets: np.ndarray, thresholds: Sequence[float], max_fg_miss: int) -> dict:
    targets = targets.astype(bool)
    rows = []
    fg_count = int(targets.sum())
    empty_count = int((~targets).sum())
    for thr in thresholds:
        keep = probs >= float(thr)
        fg_miss = int((targets & ~keep).sum())
        empty_clear = int((~targets & ~keep).sum())
        empty_keep = int((~targets & keep).sum())
        correct = int((keep == targets).sum())
        rows.append(
            {
                "threshold": float(thr),
                "fg_miss_count": fg_miss,
                "fg_miss_rate": float(fg_miss / max(1, fg_count)),
                "fg_recall": float(1.0 - fg_miss / max(1, fg_count)),
                "empty_clear_count": empty_clear,
                "empty_keep_count": empty_keep,
                "empty_clear_rate": float(empty_clear / max(1, empty_count)),
                "accuracy": float(correct / max(1, targets.size)),
            }
        )

    eligible = [r for r in rows if int(r["fg_miss_count"]) <= int(max_fg_miss)]
    if eligible:
        best = max(
            eligible,
            key=lambda r: (
                float(r["empty_clear_rate"]),
                float(r["accuracy"]),
                float(r["threshold"]),
            ),
        )
        best = dict(best)
        best["selection_feasible"] = True
    else:
        best = max(
            rows,
            key=lambda r: (
                -float(r["fg_miss_rate"]),
                float(r["empty_clear_rate"]),
                float(r["accuracy"]),
            ),
        )
        best = dict(best)
        best["selection_feasible"] = False

    best["score"] = (
        float(best["empty_clear_rate"])
        - 3.0 * float(best["fg_miss_rate"])
        + 0.01 * float(best["accuracy"])
    )
    best["fg_count"] = fg_count
    best["empty_count"] = empty_count
    best["threshold_rows"] = rows
    return best


def binary_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    targets = targets.astype(bool)
    pos = probs[targets]
    neg = probs[~targets]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, probs.size + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[targets].sum())
    return float((pos_rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


@torch.no_grad()
def evaluate_presence(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    thresholds: Sequence[float],
    max_fg_miss: int,
    use_tta: bool,
    loss_pos_weight: float,
    loss_empty_weight: float,
) -> dict:
    model.eval()
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    losses = []
    video_ids: list[str] = []
    frame_idxs: list[int] = []
    image_paths: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        targets = tensor_presence(labels)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
            weights = torch.where(
                targets > 0.5,
                torch.full_like(targets, float(loss_pos_weight)),
                torch.full_like(targets, float(loss_empty_weight)),
            )
            loss = (bce * weights).mean()

        probs = predict_presence_probs(model, images, use_amp=use_amp, use_tta=use_tta)
        losses.append(float(loss.item()))
        probs_list.append(probs.float().cpu().numpy())
        targets_list.append(targets.float().cpu().numpy())
        video_ids.extend([str(v) for v in batch["video_id"]])
        frame_idxs.extend([int(v) for v in batch["frame_idx"]])
        image_paths.extend([str(v) for v in batch["image_path"]])

    probs_np = np.concatenate(probs_list, axis=0).astype(np.float32)
    targets_np = np.concatenate(targets_list, axis=0).astype(bool)
    gate = choose_gate_threshold(probs_np, targets_np, thresholds=thresholds, max_fg_miss=max_fg_miss)
    gate["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    gate["auc"] = binary_auc(probs_np, targets_np)
    gate["prob_pos_mean"] = float(probs_np[targets_np].mean()) if bool(targets_np.any()) else float("nan")
    gate["prob_empty_mean"] = float(probs_np[~targets_np].mean()) if bool((~targets_np).any()) else float("nan")
    gate["prob_pos_min"] = float(probs_np[targets_np].min()) if bool(targets_np.any()) else float("nan")
    gate["prob_empty_max"] = float(probs_np[~targets_np].max()) if bool((~targets_np).any()) else float("nan")
    gate["probs"] = probs_np
    gate["targets"] = targets_np
    gate["video_ids"] = video_ids
    gate["frame_idxs"] = frame_idxs
    gate["image_paths"] = image_paths
    return gate


def load_seg_ckpt(ckpt_path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(args, dict):
        args = {}
    return ckpt, args


def load_seg_state_dict(model: nn.Module, ckpt_obj: dict) -> None:
    if "model_state" in ckpt_obj:
        state = ckpt_obj["model_state"]
    elif "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
    elif "state_dict" in ckpt_obj:
        state = ckpt_obj["state_dict"]
    else:
        state = ckpt_obj
    model.load_state_dict(state, strict=True)


def infer_seg_threshold(ckpt_obj: dict, fallback: float) -> float:
    if fallback >= 0.0:
        return float(fallback)
    for container_key in ("best_metrics", "val_metrics", "metrics"):
        metrics = ckpt_obj.get(container_key, {}) if isinstance(ckpt_obj, dict) else {}
        if isinstance(metrics, dict) and "val_threshold" in metrics:
            return float(metrics["val_threshold"])
    if isinstance(ckpt_obj, dict) and "best_val_threshold" in ckpt_obj:
        return float(ckpt_obj["best_val_threshold"])
    return 0.5


@torch.no_grad()
def evaluate_segmentation_gate(
    seg_ckpt_path: Path,
    samples: Sequence,
    presence_model: nn.Module,
    gate_threshold: float,
    args: argparse.Namespace,
    device: torch.device,
    logger,
) -> dict:
    ckpt, train_args = load_seg_ckpt(seg_ckpt_path)
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    image_size = tuple(int(v) for v in train_args.get("image_size", args.image_size))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    seg_threshold = infer_seg_threshold(ckpt, fallback=float(args.seg_threshold))

    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        lemonfm_ckpt=train_args.get("lemonfm_ckpt", str(args.lemonfm_ckpt)),
        lemonfm_decoder_channels=int(train_args.get("lemonfm_decoder_channels", 128)),
    ).to(device)
    load_seg_state_dict(model, ckpt)
    model.eval()
    presence_model.eval()

    ds = LabeledDataset(
        samples=samples,
        image_size=image_size,
        target_label=int(args.target_label),
        train=False,
        cache_masks=bool(args.cache_masks),
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

    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    gate_probs_list: list[np.ndarray] = []
    video_ids: list[str] = []
    frame_idxs: list[int] = []
    use_amp = bool(args.amp and device.type == "cuda")
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float()
        h, w = tuple(int(v) for v in labels.shape[-2:])
        seg_probs = predict_seg_probs(model, images, use_amp=use_amp, use_tta=bool(args.seg_tta))
        if tuple(seg_probs.shape[-2:]) != (h, w):
            seg_probs = F.interpolate(seg_probs.float(), size=(h, w), mode="bilinear", align_corners=False)
        gate_probs = predict_presence_probs(presence_model, images, use_amp=use_amp, use_tta=True)
        probs_list.append(seg_probs.float().cpu().numpy()[:, 0])
        labels_list.append(labels.numpy()[:, 0].astype(bool))
        gate_probs_list.append(gate_probs.float().cpu().numpy())
        video_ids.extend([str(v) for v in batch["video_id"]])
        frame_idxs.extend([int(v) for v in batch["frame_idx"]])
        logger.info("[seg-gate] batch %d/%d", batch_idx, len(loader))

    seg_probs_np = np.concatenate(probs_list, axis=0).astype(np.float32)
    labels_np = np.concatenate(labels_list, axis=0).astype(bool)
    gate_probs_np = np.concatenate(gate_probs_list, axis=0).astype(np.float32)
    keep = gate_probs_np >= float(gate_threshold)
    baseline_preds = seg_probs_np > float(seg_threshold)
    gated_preds = baseline_preds.copy()
    gated_preds[~keep] = False

    label_cache = build_label_cache(labels_np)
    baseline_metrics = compute_metrics(
        baseline_preds,
        label_cache,
        video_ids,
        include_per_video=True,
        include_distance=True,
    )
    gated_metrics = compute_metrics(
        gated_preds,
        label_cache,
        video_ids,
        include_per_video=True,
        include_distance=True,
    )
    gt_non_empty = label_cache.gt_non_empty
    gate_fg_miss = int((gt_non_empty & ~keep).sum())
    gate_empty_clear = int((~gt_non_empty & ~keep).sum())
    return {
        "seg_ckpt_path": str(seg_ckpt_path),
        "seg_threshold": float(seg_threshold),
        "gate_threshold": float(gate_threshold),
        "gate_fg_miss_count": gate_fg_miss,
        "gate_empty_clear_count": gate_empty_clear,
        "gate_keep_count": int(keep.sum()),
        "num_frames": int(len(keep)),
        "baseline": baseline_metrics,
        "gated": gated_metrics,
        "delta": {
            "dice_all": float(gated_metrics["dice_all"] - baseline_metrics["dice_all"]),
            "dice_fg": float(gated_metrics["dice_fg"] - baseline_metrics["dice_fg"]),
            "hd": float(gated_metrics["hd"] - baseline_metrics["hd"]),
            "asd": float(gated_metrics["asd"] - baseline_metrics["asd"]),
            "empty_fp_rate": float(gated_metrics["empty_fp_rate"] - baseline_metrics["empty_fp_rate"]),
        },
        "frames": [
            {
                "video_id": str(v),
                "frame_idx": int(idx),
                "gate_prob": float(prob),
                "gt_non_empty": bool(gt),
                "gate_keep": bool(k),
            }
            for v, idx, prob, gt, k in zip(video_ids, frame_idxs, gate_probs_np, gt_non_empty, keep)
        ],
    }


def write_history(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) else v
    return value


def strip_arrays(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        if key == "threshold_rows":
            out[key] = value
        elif isinstance(value, (np.bool_, bool)):
            out[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            out[key] = None if math.isnan(float(value)) else float(value)
        else:
            out[key] = jsonable(value)
    return out


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "args": jsonable(vars(args)),
            "epoch": int(epoch),
            "metrics": strip_arrays(metrics),
            "gate_threshold": float(metrics["threshold"]),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    if image_size[0] % 32 != 0 or image_size[1] % 32 != 0:
        raise ValueError(f"image_size must be divisible by 32, got {image_size}")

    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir, log_name="presence_gate.log")
    logger.info("Start Task3 presence gate training")
    logger.info("Args: %s", vars(args))

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type != "cuda":
        raise RuntimeError("Presence gate training is configured to require CUDA for this project.")
    torch.backends.cudnn.benchmark = True
    logger.info("Device=%s AMP=%s", device, use_amp)

    all_samples = discover_samples(args.labeled_root)
    train_samples, val_samples, train_video_ids, val_video_ids = split_train_val_by_video(
        all_samples,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )
    if int(args.max_train_samples) > 0:
        train_samples = train_samples[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_samples = val_samples[: int(args.max_val_samples)]

    train_ds = LabeledDataset(
        samples=train_samples,
        image_size=image_size,
        target_label=int(args.target_label),
        train=True,
        cache_masks=bool(args.cache_masks),
        use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )
    val_ds = LabeledDataset(
        samples=val_samples,
        image_size=image_size,
        target_label=int(args.target_label),
        train=False,
        cache_masks=bool(args.cache_masks),
        use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )
    train_presence = np.asarray([r > 0.0 for r in train_ds.sample_fg_ratio], dtype=bool)
    val_presence = np.asarray([r > 0.0 for r in val_ds.sample_fg_ratio], dtype=bool)
    logger.info(
        "Split: train=%d fg=%d empty=%d videos=%s | val=%d fg=%d empty=%d videos=%s",
        len(train_ds),
        int(train_presence.sum()),
        int((~train_presence).sum()),
        train_video_ids,
        len(val_ds),
        int(val_presence.sum()),
        int((~val_presence).sum()),
        val_video_ids,
    )
    save_json(
        out_dir / "split.json",
        {
            "train_videos": train_video_ids,
            "val_videos": val_video_ids,
            "train_frames": len(train_ds),
            "val_frames": len(val_ds),
            "train_fg_frames": int(train_presence.sum()),
            "train_empty_frames": int((~train_presence).sum()),
            "val_fg_frames": int(val_presence.sum()),
            "val_empty_frames": int((~val_presence).sum()),
        },
    )
    save_json(out_dir / "config.json", jsonable(vars(args)))

    sampler = build_presence_sampler(train_ds) if bool(args.balance_presence_sampling) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    model = LemonFMPresenceClassifier(
        pretrained_weights=args.lemonfm_ckpt,
        dropout=float(args.dropout),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    configure_trainable(model, args, logger)
    optimizer = build_optimizer(model, args)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1e9
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        set_lrs(optimizer, base_lrs, lr_factor(epoch, args))
        running = []
        epoch_start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            targets = tensor_presence(labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
                weights = torch.where(
                    targets > 0.5,
                    torch.full_like(targets, float(args.presence_pos_loss_weight)),
                    torch.full_like(targets, float(args.presence_empty_loss_weight)),
                )
                loss = (bce * weights).mean()
            scaler.scale(loss).backward()
            if float(args.grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=float(args.grad_clip_norm),
                )
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.item()))
            if step == 1 or step % max(1, int(args.print_freq)) == 0:
                logger.info(
                    "Epoch %03d step %03d/%03d loss=%.5f lr=%s",
                    epoch,
                    step,
                    len(train_loader),
                    float(np.mean(running[-max(1, int(args.print_freq)) :])),
                    ",".join(f"{float(g['lr']):.2e}" for g in optimizer.param_groups),
                )

        metrics = evaluate_presence(
            model,
            val_loader,
            device=device,
            use_amp=use_amp,
            thresholds=args.threshold_candidates,
            max_fg_miss=int(args.max_fg_miss),
            use_tta=True,
            loss_pos_weight=float(args.presence_pos_loss_weight),
            loss_empty_weight=float(args.presence_empty_loss_weight),
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(running)) if running else float("nan"),
            "val_loss": float(metrics["val_loss"]),
            "threshold": float(metrics["threshold"]),
            "score": float(metrics["score"]),
            "selection_feasible": int(bool(metrics["selection_feasible"])),
            "fg_miss_count": int(metrics["fg_miss_count"]),
            "fg_miss_rate": float(metrics["fg_miss_rate"]),
            "empty_clear_count": int(metrics["empty_clear_count"]),
            "empty_clear_rate": float(metrics["empty_clear_rate"]),
            "empty_keep_count": int(metrics["empty_keep_count"]),
            "accuracy": float(metrics["accuracy"]),
            "auc": float(metrics["auc"]),
            "prob_pos_mean": float(metrics["prob_pos_mean"]),
            "prob_empty_mean": float(metrics["prob_empty_mean"]),
            "prob_pos_min": float(metrics["prob_pos_min"]),
            "prob_empty_max": float(metrics["prob_empty_max"]),
            "elapsed_min": float((time.time() - epoch_start) / 60.0),
        }
        history.append(row)
        write_history(out_dir / "history.csv", history)
        logger.info(
            "Epoch %03d val: loss=%.5f auc=%.4f thr=%.3f feasible=%s fg_miss=%d/%d empty_clear=%d/%d acc=%.4f score=%.4f time=%.1fmin",
            epoch,
            float(row["val_loss"]),
            float(row["auc"]),
            float(row["threshold"]),
            bool(metrics["selection_feasible"]),
            int(metrics["fg_miss_count"]),
            int(metrics["fg_count"]),
            int(metrics["empty_clear_count"]),
            int(metrics["empty_count"]),
            float(row["accuracy"]),
            float(row["score"]),
            float(row["elapsed_min"]),
        )

        improved = (
            float(metrics["score"]) > best_score + 1e-9
            or (
                abs(float(metrics["score"]) - best_score) <= 1e-9
                and float(metrics["val_loss"]) < best_loss
            )
        )
        if improved:
            best_score = float(metrics["score"])
            best_loss = float(metrics["val_loss"])
            best_epoch = int(epoch)
            stale = 0
            save_checkpoint(ckpt_dir / "best.pt", model, args, epoch, metrics)
            save_json(out_dir / "best_presence_metrics.json", strip_arrays(metrics))
            logger.info("Saved new best presence gate at epoch %d", epoch)
        else:
            stale += 1

        save_checkpoint(ckpt_dir / "last.pt", model, args, epoch, metrics)
        if int(args.early_stop_patience) > 0 and stale >= int(args.early_stop_patience):
            logger.info("Early stopping at epoch %d; best_epoch=%d", epoch, best_epoch)
            break

    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    final_metrics = dict(ckpt["metrics"])
    final_metrics["best_epoch"] = int(ckpt["epoch"])
    final_metrics["total_elapsed_min"] = float((time.time() - start_time) / 60.0)

    if args.seg_ckpt_path is not None:
        seg_eval = evaluate_segmentation_gate(
            seg_ckpt_path=args.seg_ckpt_path,
            samples=val_samples,
            presence_model=model,
            gate_threshold=float(ckpt["gate_threshold"]),
            args=args,
            device=device,
            logger=logger,
        )
        save_json(out_dir / "segmentation_gate_eval.json", seg_eval)
        final_metrics["segmentation_gate_eval"] = {
            "seg_ckpt_path": seg_eval["seg_ckpt_path"],
            "seg_threshold": seg_eval["seg_threshold"],
            "gate_threshold": seg_eval["gate_threshold"],
            "baseline": seg_eval["baseline"],
            "gated": seg_eval["gated"],
            "delta": seg_eval["delta"],
            "gate_fg_miss_count": seg_eval["gate_fg_miss_count"],
            "gate_empty_clear_count": seg_eval["gate_empty_clear_count"],
        }
        logger.info(
            "Seg gate baseline: dice_all=%.4f dice_fg=%.4f hd=%.2f asd=%.2f empty_fp=%.3f",
            float(seg_eval["baseline"]["dice_all"]),
            float(seg_eval["baseline"]["dice_fg"]),
            float(seg_eval["baseline"]["hd"]),
            float(seg_eval["baseline"]["asd"]),
            float(seg_eval["baseline"]["empty_fp_rate"]),
        )
        logger.info(
            "Seg gate gated:    dice_all=%.4f dice_fg=%.4f hd=%.2f asd=%.2f empty_fp=%.3f delta_hd=%.2f delta_asd=%.2f",
            float(seg_eval["gated"]["dice_all"]),
            float(seg_eval["gated"]["dice_fg"]),
            float(seg_eval["gated"]["hd"]),
            float(seg_eval["gated"]["asd"]),
            float(seg_eval["gated"]["empty_fp_rate"]),
            float(seg_eval["delta"]["hd"]),
            float(seg_eval["delta"]["asd"]),
        )

    save_json(out_dir / "final_metrics.json", final_metrics)
    logger.info("Done. Best epoch=%d output=%s", best_epoch, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
