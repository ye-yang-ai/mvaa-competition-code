#!/usr/bin/env python3
"""Generate Task2 submission predictions from a probability ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from scipy import ndimage as ndi

from model_factory import get_model
from utils import get_device

try:
    import nibabel as nib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency nibabel. Install via: pip install nibabel") from exc

ENABLE_POSTPROCESS = True
POST_MIN_SIZE = 100
POST_KEEP_COMPONENTS = 1
POST_FILL_HOLES = True
POST_CLOSE_ITERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task2 ensemble submission predictions.")
    parser.add_argument("--ckpt-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--submission-task-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    post_group = parser.add_mutually_exclusive_group()
    post_group.add_argument("--postprocess", dest="postprocess", action="store_true", help="Enable mask post-processing.")
    post_group.add_argument("--no-postprocess", dest="postprocess", action="store_false", help="Disable mask post-processing.")
    parser.set_defaults(postprocess=ENABLE_POSTPROCESS)
    parser.add_argument("--post-min-size", type=int, default=POST_MIN_SIZE)
    parser.add_argument("--post-keep-components", type=int, default=POST_KEEP_COMPONENTS)
    parser.add_argument("--post-fill-holes", action="store_true", default=POST_FILL_HOLES)
    parser.add_argument("--post-no-fill-holes", dest="post_fill_holes", action="store_false")
    parser.add_argument("--post-close-iters", type=int, default=POST_CLOSE_ITERS)
    return parser.parse_args()


def discover_images(folder: str | Path) -> List[Dict[str, str]]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    image_files = sorted(root.glob("*-US.nii.gz"))
    if not image_files:
        raise RuntimeError(f"No image files found in: {root}")

    return [
        {
            "image": str(image_path),
            "image_path": str(image_path),
            "image_name": image_path.name,
            "case_id": image_path.name.replace("-US.nii.gz", ""),
        }
        for image_path in image_files
    ]


def get_infer_transforms(enable_spacing_resample: bool, target_spacing):
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]
    if enable_spacing_resample:
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=tuple(target_spacing),
                mode=("bilinear",),
            )
        )
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"]),
        ]
    )
    return Compose(transforms)


def build_loader(files: List[Dict[str, str]], enable_spacing_resample: bool, target_spacing, num_workers: int):
    ds = Dataset(
        files,
        transform=get_infer_transforms(
            enable_spacing_resample=enable_spacing_resample,
            target_spacing=target_spacing,
        ),
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def load_ckpt_config(ckpt_path: Path) -> Tuple[dict, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    return ckpt, train_args


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


def save_prediction_nifti(mask: np.ndarray, source_image_path: Path, save_path: Path) -> None:
    source = nib.load(str(source_image_path))
    pred_img = nib.Nifti1Image(mask.astype(np.uint8), affine=source.affine, header=source.header.copy())
    nib.save(pred_img, str(save_path))


def resize_mask_to_shape(mask: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(out_shape):
        return mask
    x = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
    y = F.interpolate(x, size=tuple(out_shape), mode="nearest")
    return y[0, 0].to(dtype=torch.uint8).cpu().numpy()


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


def normalize_weights(weights: list[float] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"--weights length ({len(weights)}) must match --ckpt-paths length ({n}).")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("--weights must sum to a positive value.")
    return [float(w) / total for w in weights]


def assert_compatible(configs: list[dict]) -> None:
    first = configs[0]
    keys = ("num_classes", "roi_size", "sw_batch_size", "enable_spacing_resample", "target_spacing")
    for cfg in configs[1:]:
        for key in keys:
            if cfg.get(key) != first.get(key):
                raise ValueError(f"Checkpoint configs differ for {key}: {first.get(key)} vs {cfg.get(key)}")


@torch.no_grad()
def main() -> int:
    args = parse_args()
    pred_dir = args.submission_task_dir
    output_json = pred_dir / "task2_predictions.json"
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    weights = normalize_weights(args.weights, len(args.ckpt_paths))
    models = []
    configs = []
    for ckpt_path in args.ckpt_paths:
        model, train_args = build_model_from_ckpt(ckpt_path, device)
        models.append(model)
        configs.append(train_args)
    assert_compatible(configs)

    cfg = configs[0]
    num_classes = int(cfg.get("num_classes", 3))
    roi_size = cfg.get("roi_size", [128, 128, 128])
    sw_batch_size = int(cfg.get("sw_batch_size", 1))
    enable_spacing_resample = bool(cfg.get("enable_spacing_resample", False))
    target_spacing = cfg.get("target_spacing", [0.5, 0.5, 0.5])

    files = discover_images(args.data_dir)
    loader = build_loader(files, enable_spacing_resample, target_spacing, args.num_workers)
    inferer = SlidingWindowInferer(
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        overlap=0.25,
        mode="gaussian",
    )

    print(f"Device: {device}")
    print(f"Checkpoints: {[str(p) for p in args.ckpt_paths]}")
    print(f"Weights: {weights}")
    print(f"Input cases: {len(files)}")
    print(f"Save labels to: {pred_dir}")
    print(
        "Postprocess: "
        f"enabled={args.postprocess}, min_size={args.post_min_size}, "
        f"keep_components={args.post_keep_components}, fill_holes={args.post_fill_holes}, "
        f"close_iters={args.post_close_iters}"
    )

    records = []
    total = len(loader)
    for idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        probs_sum = None
        for model, weight in zip(models, weights):
            logits = inferer(images, model)
            probs = torch.softmax(logits, dim=1) * weight
            probs_sum = probs if probs_sum is None else probs_sum + probs

        pred_mask = torch.argmax(probs_sum, dim=1).squeeze(0).detach().cpu().numpy()
        if int(pred_mask.max()) >= num_classes:
            raise RuntimeError(f"Unexpected label id in prediction: {int(pred_mask.max())}")

        case_id = str(batch["case_id"][0])
        image_path = Path(str(batch["image_path"][0]))
        source_img = nib.load(str(image_path))
        pred_mask = resize_mask_to_shape(pred_mask, tuple(int(x) for x in source_img.shape[:3]))
        if args.postprocess:
            pred_mask = postprocess_multiclass(
                pred_mask,
                num_classes=num_classes,
                min_size=args.post_min_size,
                keep_components=args.post_keep_components,
                fill_holes=args.post_fill_holes,
                close_iters=args.post_close_iters,
            )
        save_path = pred_dir / f"{case_id}-pred.nii.gz"
        save_prediction_nifti(pred_mask, image_path, save_path)

        records.append(
            {
                "case_id": case_id,
                "segmentation": save_path.name,
            }
        )
        print(f"[predict] {idx}/{total} -> {save_path.name}")

    result = {"cases": records}
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
