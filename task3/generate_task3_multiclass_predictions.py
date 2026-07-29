#!/usr/bin/env python3
"""Generate Task3 binary submission masks from a multiclass auxiliary model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from generate_task3_predictions import discover_images, load_state_dict, pick_device, postprocess_mask  # noqa: E402
from model_factory import get_model  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "opt" / "task3" / "t3_multiclass_res34_5class_s50" / "checkpoints" / "best.pt"
DATA_DIR = REPO_ROOT / "data" / "reference_data" / "t3_vid" / "val" / "images"
SUBMISSION_TASK_DIR = REPO_ROOT / "outputs" / "submit_multiclass" / "t3_vid"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 predictions from a multiclass auxiliary model.")
    parser.add_argument("--ckpt-path", type=Path, default=CKPT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submission-task-dir", type=Path, default=SUBMISSION_TASK_DIR)
    parser.add_argument("--video-folders", nargs="*", default=[])
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", or "cuda:N"')
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint validation threshold.")
    parser.add_argument("--line-alpha", type=float, default=None, help="Override checkpoint validation line alpha.")
    parser.add_argument("--postprocess", action="store_true", default=False)
    parser.add_argument("--post-min-area", type=int, default=0)
    parser.add_argument("--post-min-total-area", type=int, default=0)
    parser.add_argument("--post-keep-top", type=int, default=0)
    parser.add_argument("--post-fill-holes", action="store_true", default=False)
    parser.add_argument("--post-close-iters", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Debug only; 0 means all.")
    return parser.parse_args()


def load_ckpt_config(ckpt_path: Path) -> Tuple[dict, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    return ckpt, train_args


@torch.no_grad()
def predict_softmax(model, image_t: torch.Tensor, use_amp: bool, use_tta: bool) -> torch.Tensor:
    device_type = image_t.device.type
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(image_t)
    probs = torch.softmax(logits, dim=1)
    if not use_tta:
        return probs

    probs_sum = probs
    for dims in [(3,), (2,), (2, 3)]:
        x = torch.flip(image_t, dims=dims)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits_f = model(x)
        probs_f = torch.softmax(logits_f, dim=1)
        probs_sum = probs_sum + torch.flip(probs_f, dims=dims)
    return probs_sum / 4.0


def make_mask_from_probs(probs: torch.Tensor, threshold: float, line_alpha: float) -> torch.Tensor:
    mitral_prob = probs[:, 1:2]
    line_prob = probs[:, 2:3] if probs.shape[1] > 2 else torch.zeros_like(mitral_prob)
    score = mitral_prob - float(line_alpha) * line_prob
    return (score > float(threshold)).float()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    args.submission_task_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.submission_task_dir / "task3_predictions.json"

    ckpt, train_args = load_ckpt_config(args.ckpt_path)
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    lemonfm_ckpt = train_args.get("lemonfm_ckpt", None)
    lemonfm_decoder_channels = int(train_args.get("lemonfm_decoder_channels", 128))
    image_size = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    num_classes = int(train_args.get("num_classes", 5))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))

    ckpt_metrics = ckpt.get("val_metrics", {})
    ckpt_threshold = float(ckpt_metrics.get("val_threshold", 0.4))
    ckpt_line_alpha = float(ckpt_metrics.get("val_line_alpha", 0.0))
    threshold = ckpt_threshold if args.threshold is None else float(args.threshold)
    line_alpha = ckpt_line_alpha if args.line_alpha is None else float(args.line_alpha)

    files = discover_images(args.data_dir, IMAGE_EXTS, args.video_folders)
    if int(args.max_images) > 0:
        files = files[: int(args.max_images)]

    device = pick_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        lemonfm_ckpt=lemonfm_ckpt,
        lemonfm_decoder_channels=lemonfm_decoder_channels,
    ).to(device)
    load_state_dict(model, ckpt)
    model.eval()

    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Input images: {len(files)}")
    print(f"Save labels to: {args.submission_task_dir}")
    print(f"Threshold: {threshold:.4f} (ckpt={ckpt_threshold:.4f}) | line_alpha={line_alpha:.4f} (ckpt={ckpt_line_alpha:.4f})")
    print(f"TTA={args.tta} | postprocess={args.postprocess}")

    records = []
    for idx, info in enumerate(files, start=1):
        image_path = Path(info["image_path"])
        rel = Path(info["image_rel_path"])
        image_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        h, w = image_u8.shape[:2]
        image_f = image_u8.astype(np.float32) / 255.0

        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
        image_t = F.interpolate(image_t, size=image_size, mode="bilinear", align_corners=False)
        if use_imagenet_norm:
            image_t = (image_t - norm_mean) / norm_std

        probs = predict_softmax(model, image_t, use_amp=use_amp, use_tta=bool(args.tta))
        pred_small = make_mask_from_probs(probs, threshold=threshold, line_alpha=line_alpha)
        pred_orig = F.interpolate(pred_small, size=(h, w), mode="nearest")
        pred_mask = (pred_orig[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        if args.postprocess:
            pred_mask = postprocess_mask(
                pred_mask,
                min_area=int(args.post_min_area),
                keep_top=int(args.post_keep_top),
                fill_holes=bool(args.post_fill_holes),
                close_iters=int(args.post_close_iters),
            )
        if int(args.post_min_total_area) > 0 and int(pred_mask.sum()) < int(args.post_min_total_area):
            pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)

        save_path = args.submission_task_dir / rel.parent / f"{image_path.stem}_label_bin.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").save(save_path)
        records.append(
            {
                "case_id": info["case_id"],
                "segmentation": save_path.relative_to(output_json.parent).as_posix(),
            }
        )
        print(f"[predict] {idx}/{len(files)} -> {save_path.name}")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)
    print(f"Saved json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
