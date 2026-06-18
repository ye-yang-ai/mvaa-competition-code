#!/usr/bin/env python3
"""Probe MedSAM2 image encoder features for MVAA Task3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
MEDSAM2_ROOT = REPO_ROOT / "external" / "MedSAM2"
DEFAULT_CFG = "configs/sam2.1_hiera_t512.yaml"
DEFAULT_CKPT = MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "data" / "t3_vid" / "train"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe MedSAM2 image_encoder output shapes.")
    parser.add_argument("--config", type=str, default=DEFAULT_CFG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    return parser.parse_args()


def pick_device(mode: str, gpu_id: int) -> torch.device:
    mode = str(mode).lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{int(gpu_id)}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cpu")


def discover_first_task3_frame(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"image root not found: {root}")
    for path in sorted(root.rglob("*.png")):
        name = path.name.lower()
        if name.endswith("_label_bin.png") or "_png_label_vis" in name:
            continue
        return path
    raise RuntimeError(f"No Task3 frame png found under: {root}")


def load_image_tensor(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def shape_str(x: torch.Tensor) -> str:
    return "x".join(str(v) for v in tuple(x.shape))


def main() -> int:
    args = parse_args()

    if str(MEDSAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDSAM2_ROOT))

    from sam2.build_sam import build_sam2

    if not args.ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    image_path = args.image_path or discover_first_task3_frame(args.image_root)
    device = pick_device(args.device, args.gpu_id)
    use_amp = bool(args.amp and device.type == "cuda")

    print(f"MedSAM2 root: {MEDSAM2_ROOT}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Image: {image_path}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Device: {device} | AMP={use_amp}")

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    model = build_sam2(
        config_file=args.config,
        ckpt_path=str(args.ckpt),
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )
    image_encoder = model.image_encoder.eval()
    total, trainable = count_params(image_encoder)
    print(f"Image encoder params: total={total:,} trainable={trainable:,}")
    print(f"Trunk channels: {getattr(image_encoder.trunk, 'channel_list', 'unknown')}")
    print(f"Neck channels: {getattr(image_encoder.neck, 'backbone_channel_list', 'unknown')}")
    print(f"Neck d_model: {getattr(image_encoder.neck, 'd_model', 'unknown')}")
    print(f"Scalp: {getattr(image_encoder, 'scalp', 'unknown')}")

    image_t = load_image_tensor(image_path, int(args.image_size), device)
    print(f"Input tensor: shape={shape_str(image_t)} dtype={image_t.dtype}")

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = image_encoder(image_t)

    print(f"Output keys: {sorted(out.keys())}")
    vision_features = out["vision_features"]
    print(f"vision_features: shape={shape_str(vision_features)} dtype={vision_features.dtype}")

    backbone_fpn = out["backbone_fpn"]
    print(f"backbone_fpn levels: {len(backbone_fpn)}")
    for idx, feat in enumerate(backbone_fpn):
        stride_h = int(args.image_size) // int(feat.shape[-2])
        stride_w = int(args.image_size) // int(feat.shape[-1])
        print(
            f"  level {idx}: shape={shape_str(feat)} dtype={feat.dtype} "
            f"stride=({stride_h},{stride_w}) min={float(feat.min()):.4f} max={float(feat.max()):.4f}"
        )

    pos = out.get("vision_pos_enc", [])
    print(f"vision_pos_enc levels: {len(pos)}")
    for idx, pe in enumerate(pos):
        print(f"  pos {idx}: shape={shape_str(pe)} dtype={pe.dtype}")

    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
        print(f"CUDA peak memory: allocated={allocated:.3f} GiB reserved={reserved:.3f} GiB")

    print("Probe OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
