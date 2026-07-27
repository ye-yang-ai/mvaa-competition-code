#!/usr/bin/env python3
"""Task3 multiclass auxiliary segmentation training.

The submitted mask is still binary label 10. Extra classes are used only to
teach the model dangerous negatives such as wire/suture and instruments.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    MultiClassLabeledDataset,
    Sample,
    discover_samples,
    split_train_val_by_video,
)
from model_factory import get_model
from utils import (
    MetricRefs,
    ensure_dir,
    get_device,
    metric_quality_weighted,
    save_json,
    seed_everything,
    setup_logger,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task3 5-class auxiliary segmentation training")
    parser.add_argument("--labeled-root", type=str, default=str(REPO_ROOT / "data" / "reference_data" / "t3_vid" / "train"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "opt" / "task3" / "t3_multiclass_res34_5class_s50"))

    parser.add_argument("--arch", type=str, default="unetplusplus", choices=["unet", "unetplusplus", "fpn", "deeplabv3plus"])
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", choices=["none", "imagenet"])
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--image-size", type=int, nargs=2, default=[448, 800], help="H W")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--ce-loss-weight", type=float, default=0.5)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--class-weights", type=float, nargs="+", default=[0.3, 2.0, 2.0, 1.2, 1.2])

    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--score-dsc-weight", type=float, default=0.6)
    parser.add_argument("--score-hd-weight", type=float, default=0.2)
    parser.add_argument("--score-asd-weight", type=float, default=0.2)
    parser.add_argument("--score-hd-ref", type=float, default=20.0)
    parser.add_argument("--score-asd-ref", type=float, default=3.0)
    parser.add_argument("--score-dice-mode", type=str, default="all", choices=["all", "fg"])

    parser.add_argument("--threshold-candidates", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--line-alpha-candidates", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75])
    parser.add_argument("--val-tta", action="store_true", default=True)
    parser.add_argument("--no-val-tta", action="store_false", dest="val_tta")

    parser.add_argument("--use-imagenet-norm", action="store_true", default=True)
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.add_argument("--cache-labels", action="store_true", default=True)
    parser.add_argument("--no-cache-labels", action="store_false", dest="cache_labels")

    parser.add_argument("--use-balanced-sampling", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampling", action="store_false", dest="use_balanced_sampling")
    parser.add_argument("--fg-weight", type=float, default=2.0)
    parser.add_argument("--line-weight", type=float, default=3.0)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--early-stop-patience", type=int, default=35)
    return parser.parse_args()


def dice_from_preds(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6, ignore_empty_gt: bool = False) -> float:
    inter = (preds * labels).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    gt_sum = labels.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    if ignore_empty_gt:
        valid = gt_sum > 0
        if bool(valid.any()):
            dice = dice[valid]
        else:
            return 0.0
    return float(dice.mean().item())


def _nanmean_to_float(x: torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(torch.nanmean(x).item())
    return float(x)


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: Sequence[float],
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        weight_t = torch.tensor(list(class_weights), dtype=torch.float32)
        if int(weight_t.numel()) != self.num_classes:
            raise ValueError(f"class_weights length must equal num_classes={self.num_classes}, got {weight_t.numel()}")
        self.register_buffer("class_weights", weight_t)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets.long(), weight=self.class_weights)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice[1:].mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


@torch.no_grad()
def predict_softmax(model: nn.Module, images: torch.Tensor, use_amp: bool, use_tta: bool) -> torch.Tensor:
    device_type = images.device.type
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(images)
    probs = torch.softmax(logits, dim=1)
    if not use_tta:
        return probs

    probs_sum = probs
    for dims in [(3,), (2,), (2, 3)]:
        x = torch.flip(images, dims=dims)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits_f = model(x)
        probs_f = torch.softmax(logits_f, dim=1)
        probs_sum = probs_sum + torch.flip(probs_f, dims=dims)
    return probs_sum / 4.0


def make_line_aware_preds(probs: torch.Tensor, threshold: float, line_alpha: float) -> torch.Tensor:
    mitral = probs[:, 1:2]
    line = probs[:, 2:3] if probs.shape[1] > 2 else torch.zeros_like(mitral)
    score = mitral - float(line_alpha) * line
    return (score > float(threshold)).float()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold_candidates: Sequence[float],
    line_alpha_candidates: Sequence[float],
    use_tta: bool,
    score_dice_mode: str,
    refs: MetricRefs,
    score_weights: tuple[float, float, float],
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    steps = 0
    probs_list = []
    labels_list = []
    line_labels_list = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        probs = predict_softmax(model, images, use_amp=use_amp, use_tta=use_tta)
        loss_sum += float(loss.item())
        steps += 1
        probs_list.append(probs.float().cpu())
        labels_list.append(batch["mitral"].float().cpu())
        line_labels_list.append(batch["line"].float().cpu())

    all_probs = torch.cat(probs_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    all_line_labels = torch.cat(line_labels_list, dim=0)
    gt_non_empty = all_labels.flatten(1).sum(dim=1) > 0

    best = None
    for alpha in line_alpha_candidates:
        for thr in threshold_candidates:
            preds = make_line_aware_preds(all_probs, threshold=float(thr), line_alpha=float(alpha))
            dice_fg = dice_from_preds(preds, all_labels, ignore_empty_gt=True)
            dice_all = dice_from_preds(preds, all_labels, ignore_empty_gt=False)
            pred_non_empty = preds.flatten(1).sum(dim=1) > 0
            valid_dist_mask = pred_non_empty & gt_non_empty
            valid_dist_cases = int(valid_dist_mask.sum().item())
            if valid_dist_cases > 0:
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
                for i in range(0, valid_dist_cases, 8):
                    hd_metric(y_pred=preds[valid_dist_mask][i : i + 8], y=all_labels[valid_dist_mask][i : i + 8])
                    asd_metric(y_pred=preds[valid_dist_mask][i : i + 8], y=all_labels[valid_dist_mask][i : i + 8])
                hd = _nanmean_to_float(hd_metric.aggregate())
                asd = _nanmean_to_float(asd_metric.aggregate())
                hd_metric.reset()
                asd_metric.reset()
            else:
                hd = float("nan")
                asd = float("nan")

            dice_for_score = dice_all if str(score_dice_mode) == "all" else dice_fg
            score = metric_quality_weighted(
                dsc=dice_for_score,
                hd=hd,
                asd=asd,
                refs=refs,
                dsc_weight=score_weights[0],
                hd_weight=score_weights[1],
                asd_weight=score_weights[2],
            )["score"]
            row = {
                "val_threshold": float(thr),
                "val_line_alpha": float(alpha),
                "val_dice": float(dice_fg),
                "val_dice_fg": float(dice_fg),
                "val_dice_all": float(dice_all),
                "val_hd": float(hd),
                "val_asd": float(asd),
                "val_score": float(score),
                "val_valid_dist_cases": int(valid_dist_cases),
                "val_pred_pos_ratio": float(preds.mean().item()),
                "val_gt_pos_ratio": float(all_labels.mean().item()),
                "val_line_gt_pos_ratio": float(all_line_labels.mean().item()),
                "val_empty_fp_rate": float(pred_non_empty[~gt_non_empty].float().mean().item()) if bool((~gt_non_empty).any()) else float("nan"),
            }
            if best is None or row["val_score"] > best["val_score"]:
                best = row

    assert best is not None
    best["val_loss"] = float(loss_sum / max(1, steps))
    return best


def build_sampling_weights(ds: MultiClassLabeledDataset, fg_weight: float, line_weight: float) -> torch.Tensor:
    weights = []
    for mitral_ratio, line_ratio in zip(ds.sample_mitral_ratio, ds.sample_line_ratio):
        w = 1.0
        if mitral_ratio > 0:
            w += float(fg_weight)
        if line_ratio > 0:
            w += float(line_weight)
        weights.append(w)
    return torch.as_tensor(weights, dtype=torch.double)


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, args: argparse.Namespace, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "val_metrics": metrics,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))
    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir)

    device = get_device()
    use_amp = bool(args.amp) and device.type == "cuda"
    logger.info("Device: %s | amp=%s", device, use_amp)

    samples = discover_samples(args.labeled_root)
    train_samples, val_samples, train_videos, val_videos = split_train_val_by_video(
        samples,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )
    if int(args.max_train_samples) > 0:
        train_samples = train_samples[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_samples = val_samples[: int(args.max_val_samples)]

    image_size = tuple(int(v) for v in args.image_size)
    train_ds = MultiClassLabeledDataset(
        samples=train_samples,
        image_size=image_size,
        num_classes=int(args.num_classes),
        train=True,
        cache_labels=bool(args.cache_labels),
        use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )
    val_ds = MultiClassLabeledDataset(
        samples=val_samples,
        image_size=image_size,
        num_classes=int(args.num_classes),
        train=False,
        cache_labels=bool(args.cache_labels),
        use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )

    sampler = None
    if bool(args.use_balanced_sampling):
        weights = build_sampling_weights(train_ds, fg_weight=float(args.fg_weight), line_weight=float(args.line_weight))
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(int(args.seed)),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
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
        num_workers=min(2, int(args.num_workers)),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    encoder_weights = None if str(args.encoder_weights).lower() == "none" else str(args.encoder_weights)
    model = get_model(
        arch=str(args.arch),
        encoder_name=str(args.encoder_name),
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=int(args.num_classes),
    ).to(device)
    loss_fn = CrossEntropyDiceLoss(
        num_classes=int(args.num_classes),
        class_weights=args.class_weights,
        ce_weight=float(args.ce_loss_weight),
        dice_weight=float(args.dice_loss_weight),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    warmup_epochs = max(0, min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)))
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(args.epochs) - warmup_epochs),
                    eta_min=float(args.min_lr),
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=float(args.min_lr),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    save_json(out_dir / "config.json", vars(args))
    save_json(
        out_dir / "split.json",
        {
            "labeled_root": str(args.labeled_root),
            "train_videos": train_videos,
            "val_videos": val_videos,
            "train_samples": [s.image_path.as_posix() for s in train_samples],
            "val_samples": [s.image_path.as_posix() for s in val_samples],
        },
    )

    logger.info(
        "Data: all=%d train=%d val=%d | train_videos=%s | val_videos=%s | mitral_ratio=%.6f line_ratio=%.6f",
        len(samples),
        len(train_ds),
        len(val_ds),
        train_videos,
        val_videos,
        float(sum(train_ds.sample_mitral_ratio) / max(1, len(train_ds.sample_mitral_ratio))),
        float(sum(train_ds.sample_line_ratio) / max(1, len(train_ds.sample_line_ratio))),
    )
    logger.info(
        "Model: arch=%s encoder=%s weights=%s classes=%d image_size=%s class_weights=%s",
        args.arch,
        args.encoder_name,
        args.encoder_weights,
        int(args.num_classes),
        image_size,
        args.class_weights,
    )

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_mitral_dice",
                "val_loss",
                "val_dice",
                "val_dice_all",
                "val_hd",
                "val_asd",
                "val_threshold",
                "val_line_alpha",
                "val_score",
                "lr",
                "epoch_sec",
            ]
        )

    refs = MetricRefs(hd_ref=float(args.score_hd_ref), asd_ref=float(args.score_asd_ref))
    score_weights = (float(args.score_dsc_weight), float(args.score_hd_weight), float(args.score_asd_weight))
    best_score = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    no_improve = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        dice_sum = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            mitral = batch["mitral"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                if float(args.grad_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits.detach(), dim=1)
                preds = (probs[:, 1:2] > 0.5).float()
                dice_sum += dice_from_preds(preds, mitral, ignore_empty_gt=True)
            loss_sum += float(loss.item())
            steps += 1

            if int(args.print_freq) > 0 and step % int(args.print_freq) == 0:
                logger.info(
                    "Epoch %d step %d/%d | loss=%.4f mitral_dice=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    loss_sum / max(1, steps),
                    dice_sum / max(1, steps),
                )

        scheduler.step()
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            threshold_candidates=args.threshold_candidates,
            line_alpha_candidates=args.line_alpha_candidates,
            use_tta=bool(args.val_tta),
            score_dice_mode=str(args.score_dice_mode),
            refs=refs,
            score_weights=score_weights,
        )
        score = float(val_metrics["val_score"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            no_improve = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, args, val_metrics)
        else:
            no_improve += 1

        if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0 or epoch == int(args.epochs)):
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, epoch, args, val_metrics)
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, args, val_metrics)

        epoch_sec = time.time() - epoch_start
        train_loss = loss_sum / max(1, steps)
        train_dice = dice_sum / max(1, steps)
        lr = float(optimizer.param_groups[0]["lr"])
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_dice:.6f}",
                    f"{val_metrics['val_loss']:.6f}",
                    f"{val_metrics['val_dice']:.6f}",
                    f"{val_metrics['val_dice_all']:.6f}",
                    f"{val_metrics['val_hd']:.6f}",
                    f"{val_metrics['val_asd']:.6f}",
                    f"{val_metrics['val_threshold']:.4f}",
                    f"{val_metrics['val_line_alpha']:.4f}",
                    f"{score:.6f}",
                    f"{lr:.8g}",
                    f"{epoch_sec:.2f}",
                ]
            )

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_dice=%.4f | val_loss=%.4f dice=%.4f all_dice=%.4f "
            "hd=%s asd=%s thr=%.2f alpha=%.2f score=%.4f best=%.4f(epoch=%d) | %.1fs",
            epoch,
            int(args.epochs),
            train_loss,
            train_dice,
            float(val_metrics["val_loss"]),
            float(val_metrics["val_dice"]),
            float(val_metrics["val_dice_all"]),
            "nan" if math.isnan(float(val_metrics["val_hd"])) else f"{float(val_metrics['val_hd']):.4f}",
            "nan" if math.isnan(float(val_metrics["val_asd"])) else f"{float(val_metrics['val_asd']):.4f}",
            float(val_metrics["val_threshold"]),
            float(val_metrics["val_line_alpha"]),
            score,
            best_score,
            best_epoch,
            epoch_sec,
        )

        if int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
            logger.info("Early stop at epoch %d after %d non-improving epochs.", epoch, no_improve)
            break

    save_json(
        out_dir / "summary.json",
        {
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "output_dir": str(out_dir),
        },
    )
    logger.info("Done. best_epoch=%d best_score=%.6f best_metrics=%s", best_epoch, best_score, best_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
