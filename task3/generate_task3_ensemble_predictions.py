#!/usr/bin/env python3
"""Generate Task3 predictions from a two-model probability ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from generate_task3_predictions import discover_images, load_state_dict, predict_probs  # noqa: E402
from model_factory import get_model  # noqa: E402

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 ensemble submission predictions.")
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/val/images")
    parser.add_argument("--submission-task-dir", type=Path, required=True)
    parser.add_argument("--video-folders", nargs="*", default=[])
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


def build_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
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
    return model, train_args


@torch.no_grad()
def main() -> int:
    args = parse_args()
    output_json = args.submission_task_dir / "task3_predictions.json"
    args.submission_task_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    weight_a = float(args.weight_a)
    weight_b = 1.0 - weight_a

    model_a, train_args_a = build_model(args.ckpt_a, device=device)
    model_b, train_args_b = build_model(args.ckpt_b, device=device)
    size_a = tuple(int(v) for v in train_args_a.get("image_size", [448, 800]))
    size_b = tuple(int(v) for v in train_args_b.get("image_size", [448, 800]))

    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    files = discover_images(args.data_dir, IMAGE_EXTS, args.video_folders)
    print(f"Device: {device} AMP={use_amp} TTA={args.tta}")
    print(f"Checkpoint A: {args.ckpt_a} weight={weight_a:.4f} size={size_a}")
    print(f"Checkpoint B: {args.ckpt_b} weight={weight_b:.4f} size={size_b}")
    print(f"Threshold: {float(args.threshold):.4f}")
    print(f"Input images: {len(files)}")
    print(f"Save labels to: {args.submission_task_dir}")

    records: List[Dict[str, str]] = []
    for idx, info in enumerate(files, start=1):
        image_path = Path(info["image_path"])
        rel = Path(info["image_rel_path"])
        image_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        h, w = image_u8.shape[:2]
        image_f = image_u8.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
        image_t = (image_t - norm_mean) / norm_std

        x_a = F.interpolate(image_t, size=size_a, mode="bilinear", align_corners=False)
        x_b = F.interpolate(image_t, size=size_b, mode="bilinear", align_corners=False)
        pa = predict_probs(model_a, x_a, use_amp=use_amp, use_tta=bool(args.tta))
        pb = predict_probs(model_b, x_b, use_amp=use_amp, use_tta=bool(args.tta))
        pa = F.interpolate(pa.float(), size=(h, w), mode="bilinear", align_corners=False)
        pb = F.interpolate(pb.float(), size=(h, w), mode="bilinear", align_corners=False)
        probs = weight_a * pa + weight_b * pb
        pred_mask = (probs[0, 0].detach().cpu().numpy() > float(args.threshold)).astype(np.uint8)

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
