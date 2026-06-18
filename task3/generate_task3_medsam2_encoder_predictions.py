#!/usr/bin/env python3
"""Generate Task3 predictions from the MedSAM2 encoder Stage A model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from medsam2_backbone import Task3MedSAM2EncoderSeg  # noqa: E402
from medsam2_backbone.build_encoder import DEFAULT_ENCODER_CFG, DEFAULT_ENCODER_CKPT  # noqa: E402


CKPT_PATH = REPO_ROOT / "outputs" / "medsam2_stageA" / "task3_frozen" / "checkpoints" / "best.pt"
DATA_DIR = REPO_ROOT / "data" / "t3_vid" / "val" / "images"
SUBMISSION_TASK_DIR = REPO_ROOT / "outputs" / "medsam2_stageA" / "submission" / "t3_vid"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
VIDEO_FOLDERS = []
USE_TTA = True
AMP = True
DEVICE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 MedSAM2 encoder submission predictions.")
    parser.add_argument("--ckpt-path", type=Path, default=CKPT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submission-task-dir", type=Path, default=SUBMISSION_TASK_DIR)
    parser.add_argument("--video-folders", nargs="*", default=VIDEO_FOLDERS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEVICE)
    parser.add_argument("--tta", action="store_true", default=USE_TTA)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--amp", action="store_true", default=AMP)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    return parser.parse_args()


def discover_images(folder: str | Path, exts: List[str], video_folders: List[str]) -> List[Dict[str, str]]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    exts_set = {e.lower() for e in exts}
    target_dirs = [root / x for x in video_folders] if video_folders else [root]
    for d in target_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Video folder not found under DATA_DIR: {d}")

    files: List[Path] = []
    for base in target_dirs:
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts_set:
                name_l = p.name.lower()
                if name_l.endswith("_label_bin.png"):
                    continue
                if "_png_label_vis" in name_l:
                    continue
                files.append(p)
    if not files:
        raise RuntimeError(f"No image files found in: {root}")

    return [
        {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_rel_path": image_path.relative_to(root).as_posix(),
            "case_id": image_path.stem,
        }
        for image_path in files
    ]


def load_ckpt_config(ckpt_path: Path) -> Tuple[dict, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    return ckpt, train_args


def pick_device(device_arg: str) -> torch.device:
    mode = str(device_arg).lower().strip()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def predict_probs(model, image_t: torch.Tensor, use_amp: bool, use_tta: bool) -> torch.Tensor:
    device_type = image_t.device.type
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(image_t)
    probs = torch.sigmoid(logits)

    if not use_tta:
        return probs

    probs_sum = probs
    for dims in [(3,), (2,), (2, 3)]:
        x = torch.flip(image_t, dims=dims)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits_f = model(x)
        probs_f = torch.sigmoid(logits_f)
        probs_sum = probs_sum + torch.flip(probs_f, dims=dims)
    return probs_sum / 4.0


@torch.no_grad()
def main() -> int:
    args = parse_args()
    ckpt_path = args.ckpt_path
    data_dir = args.data_dir
    pred_dir = args.submission_task_dir
    output_json = args.submission_task_dir / "task3_predictions.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt, train_args = load_ckpt_config(ckpt_path)
    image_size = tuple(int(v) for v in train_args.get("image_size", [512, 512]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    threshold = float(ckpt.get("val_metrics", {}).get("val_threshold", 0.5))
    encoder_cfg = str(train_args.get("encoder_cfg", DEFAULT_ENCODER_CFG))
    encoder_ckpt = str(train_args.get("encoder_ckpt", DEFAULT_ENCODER_CKPT))
    decoder_channels = int(train_args.get("decoder_channels", 128))
    freeze_encoder = bool(train_args.get("freeze_encoder", True))

    files = discover_images(data_dir, IMAGE_EXTS, args.video_folders)
    device = pick_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"

    model = Task3MedSAM2EncoderSeg(
        encoder_cfg=encoder_cfg,
        encoder_ckpt=encoder_ckpt,
        decoder_channels=decoder_channels,
        freeze_encoder=freeze_encoder,
        device=device,
    ).to(device)
    load_state_dict(model, ckpt)
    model.eval()

    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input images: {len(files)}")
    print(f"Save labels to: {pred_dir}")
    print(f"Image size: {image_size}")
    print(f"Threshold: {threshold:.4f} | TTA={args.tta}")
    print(f"Video folders: {args.video_folders if args.video_folders else '[ALL]'}")

    records = []
    total = len(files)
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

        probs = predict_probs(model=model, image_t=image_t, use_amp=use_amp, use_tta=bool(args.tta))
        pred_small = (probs > threshold).float()
        pred_orig = F.interpolate(pred_small, size=(h, w), mode="nearest")
        pred_mask = (pred_orig[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        save_path = pred_dir / rel.parent / f"{image_path.stem}_label_bin.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").save(save_path)

        records.append(
            {
                "case_id": info["case_id"],
                "segmentation": save_path.relative_to(output_json.parent).as_posix(),
            }
        )
        print(f"[predict] {idx}/{total} -> {save_path.name}")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)

    print(f"Saved json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
