#!/usr/bin/env python3
"""Generate Task3 labels and a submission-ready JSON for Codabench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model_factory import get_model  # noqa: E402

# ===== Config (edit here) =====
CKPT_PATH = THIS_DIR / "runs" / "semi_baseline_default" / "checkpoints" / "best.pt"
DATA_DIR = REPO_ROOT / "data" / "t3_vid" / "val" / "images"
SUBMISSION_TASK_DIR = THIS_DIR.parent / "submission" / "t3_vid"
PRED_DIR = SUBMISSION_TASK_DIR
OUTPUT_JSON = SUBMISSION_TASK_DIR / "task3_predictions.json"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
VIDEO_FOLDERS = []  # Empty means infer all folders/images under DATA_DIR.
USE_TTA = True
AMP = True
DEVICE = "auto"  # "auto" | "cuda" | "cpu"
# ==============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 submission predictions.")
    parser.add_argument("--ckpt-path", type=Path, default=CKPT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--submission-task-dir", type=Path, default=SUBMISSION_TASK_DIR)
    parser.add_argument("--video-folders", nargs="*", default=VIDEO_FOLDERS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEVICE)
    parser.add_argument("--tta", action="store_true", default=USE_TTA)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument("--amp", action="store_true", default=AMP)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint validation threshold")
    parser.add_argument("--postprocess", action="store_true", default=False)
    parser.add_argument("--post-min-area", type=int, default=0)
    parser.add_argument("--post-keep-top", type=int, default=0, help="0 means keep all components after area filtering")
    parser.add_argument("--post-fill-holes", action="store_true", default=False)
    parser.add_argument("--post-close-iters", type=int, default=0)
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


def postprocess_mask(
    mask: np.ndarray,
    min_area: int,
    keep_top: int,
    fill_holes: bool,
    close_iters: int,
) -> np.ndarray:
    out = mask.astype(bool)
    labeled, num_components = ndi.label(out)
    if num_components > 0:
        areas = np.bincount(labeled.ravel())
        keep = np.zeros(num_components + 1, dtype=bool)
        comp_ids = np.arange(1, num_components + 1)
        comp_areas = areas[1:]
        valid = comp_ids[comp_areas >= int(min_area)]
        if int(keep_top) > 0 and valid.size > 0:
            valid_areas = areas[valid]
            order = np.argsort(valid_areas)[::-1][: int(keep_top)]
            valid = valid[order]
        keep[valid] = True
        out = keep[labeled]
    else:
        out = np.zeros_like(out, dtype=bool)

    if fill_holes:
        out = ndi.binary_fill_holes(out)
    if int(close_iters) > 0:
        out = ndi.binary_closing(out, structure=np.ones((3, 3), dtype=bool), iterations=int(close_iters))
    return out.astype(np.uint8)


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
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None

    image_size = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    target_label = int(train_args.get("target_label", 10))
    ckpt_threshold = float(ckpt.get("val_metrics", {}).get("val_threshold", 0.5))
    threshold = ckpt_threshold if args.threshold is None else float(args.threshold)

    files = discover_images(data_dir, IMAGE_EXTS, args.video_folders)
    device = pick_device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"

    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)
    load_state_dict(model, ckpt)
    model.eval()

    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Input images: {len(files)}")
    print(f"Save labels to: {pred_dir}")
    print(f"Threshold: {threshold:.4f} (ckpt={ckpt_threshold:.4f}) | TTA={args.tta}")
    print(
        "Postprocess: "
        f"enabled={args.postprocess}, min_area={args.post_min_area}, "
        f"keep_top={args.post_keep_top}, fill_holes={args.post_fill_holes}, "
        f"close_iters={args.post_close_iters}"
    )
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
        if args.postprocess:
            pred_mask = postprocess_mask(
                pred_mask,
                min_area=int(args.post_min_area),
                keep_top=int(args.post_keep_top),
                fill_holes=bool(args.post_fill_holes),
                close_iters=int(args.post_close_iters),
            )

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

    # Keep submission JSON minimal and evaluator-friendly.
    result = {"cases": records}

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
