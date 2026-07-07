#!/usr/bin/env python3
"""Generate Task2 labels and a submission-ready JSON for Codabench."""

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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

# ===== Config (edit here) =====
CKPT_PATH = THIS_DIR / "runs" / "full_supervised2" / "checkpoints" / "best_model.pt"
DATA_DIR = REPO_ROOT / "data" / "t2_tee" / "val" / "images"
SUBMISSION_TASK_DIR = THIS_DIR.parent / "submission" / "t2_tee"
PRED_DIR = SUBMISSION_TASK_DIR
OUTPUT_JSON = SUBMISSION_TASK_DIR / "task2_predictions.json"
NUM_WORKERS = 0
ENABLE_POSTPROCESS = True
POST_MIN_SIZE = 100
POST_KEEP_COMPONENTS = 1
POST_FILL_HOLES = True
POST_CLOSE_ITERS = 0
# ==============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task2 submission predictions.")
    parser.add_argument("--ckpt-path", type=Path, default=CKPT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submission-task-dir", type=Path, default=SUBMISSION_TASK_DIR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default="", help="Optional torch device, e.g. cuda:4")
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


def save_prediction_nifti(mask: np.ndarray, source_image_path: Path, save_path: Path) -> None:
    source = nib.load(str(source_image_path))
    pred_img = nib.Nifti1Image(mask.astype(np.uint8), affine=source.affine, header=source.header.copy())
    nib.save(pred_img, str(save_path))


def load_ckpt_config(ckpt_path: Path) -> Tuple[dict, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    return ckpt, train_args


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


@torch.no_grad()
def main() -> int:
    args = parse_args()
    ckpt_path = args.ckpt_path
    data_dir = args.data_dir
    pred_dir = args.submission_task_dir
    output_json = args.submission_task_dir / "task2_predictions.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt, train_args = load_ckpt_config(ckpt_path)
    num_classes = int(train_args.get("num_classes", 3))
    model_name = str(train_args.get("model", "unet3d"))
    model_size = str(train_args.get("model_size", "large"))
    roi_size = train_args.get("roi_size", [128, 128, 128])
    sw_batch_size = int(train_args.get("sw_batch_size", 8))
    enable_spacing_resample = bool(train_args.get("enable_spacing_resample", False))
    target_spacing = train_args.get("target_spacing", [0.5, 0.5, 0.5])

    files = discover_images(data_dir)
    loader = build_loader(files, enable_spacing_resample, target_spacing, args.num_workers)

    device = get_device(args.device)
    model = get_model(
        name=model_name,
        model_size=model_size,
        in_channels=1,
        out_channels=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        overlap=0.25,
        mode="gaussian",
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
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
        logits = inferer(images, model)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

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
