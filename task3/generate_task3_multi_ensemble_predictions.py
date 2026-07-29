#!/usr/bin/env python3
"""Generate Task3 predictions from a weighted multi-model probability ensemble."""

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
from generate_task3_ensemble_predictions import build_model  # noqa: E402
from generate_task3_predictions import discover_images, load_presence_gate, pick_device, predict_probs  # noqa: E402

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task3 weighted multi-model ensemble predictions.")
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/val/images")
    parser.add_argument("--submission-task-dir", type=Path, required=True)
    parser.add_argument("--video-folders", nargs="*", default=[])
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", or "cuda:N"')
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")
    parser.add_argument(
        "--presence-gate-ckpt",
        type=Path,
        default=None,
        help="Optional frame-level presence gate checkpoint. Applies only to selected model branches.",
    )
    parser.add_argument("--presence-gate-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--presence-gate-model-indices",
        type=int,
        nargs="*",
        default=[],
        help="0-based model indices gated by presence; defaults to the last model when a gate is provided.",
    )
    parser.add_argument("--presence-gate-tta", action="store_true", default=True)
    parser.add_argument("--no-presence-gate-tta", action="store_false", dest="presence_gate_tta")
    return parser.parse_args()


def normalized_weights(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("--weights must sum to a positive value.")
    return [float(w) / total for w in weights]


def resolve_presence_gate_indices(args: argparse.Namespace, n_models: int) -> set[int]:
    if args.presence_gate_ckpt is None:
        return set()
    raw_indices = list(args.presence_gate_model_indices)
    if not raw_indices:
        raw_indices = [n_models - 1]
    indices = {int(i) for i in raw_indices}
    bad = sorted(i for i in indices if i < 0 or i >= n_models)
    if bad:
        raise ValueError(f"--presence-gate-model-indices out of range for {n_models} models: {bad}")
    return indices


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if len(args.ckpts) != len(args.weights):
        raise ValueError(f"--ckpts count ({len(args.ckpts)}) must match --weights count ({len(args.weights)}).")

    output_json = args.submission_task_dir / "task3_predictions.json"
    args.submission_task_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    weights = normalized_weights([float(w) for w in args.weights])
    gated_model_indices = resolve_presence_gate_indices(args, n_models=len(args.ckpts))

    models = []
    sizes = []
    for ckpt_path in args.ckpts:
        model, train_args = build_model(ckpt_path, device=device)
        models.append(model)
        sizes.append(tuple(int(v) for v in train_args.get("image_size", [448, 800])))

    presence_gate = None
    presence_gate_threshold = None
    presence_predict_fn = None
    presence_gate_size = None
    if args.presence_gate_ckpt is not None:
        presence_gate, presence_gate_threshold, presence_predict_fn = load_presence_gate(
            args.presence_gate_ckpt,
            device=device,
            threshold_override=float(args.presence_gate_threshold),
        )
        first_gated = min(gated_model_indices)
        presence_gate_size = sizes[first_gated]

    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    files = discover_images(args.data_dir, IMAGE_EXTS, args.video_folders)
    print(f"Device: {device} AMP={use_amp} TTA={args.tta}")
    for idx, (ckpt_path, weight, size) in enumerate(zip(args.ckpts, weights, sizes), start=1):
        print(f"Checkpoint {idx}: {ckpt_path} weight={weight:.4f} size={size}")
    print(f"Threshold: {float(args.threshold):.4f}")
    if presence_gate is not None:
        print(
            "Presence gate: "
            f"ckpt={args.presence_gate_ckpt}, threshold={presence_gate_threshold:.4f}, "
            f"model_indices={sorted(gated_model_indices)}, size={presence_gate_size}, "
            f"TTA={args.presence_gate_tta}"
        )
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

        gate_keep = True
        gate_prob = None
        if presence_gate is not None:
            gate_x = F.interpolate(image_t, size=presence_gate_size, mode="bilinear", align_corners=False)
            gate_probs = presence_predict_fn(
                presence_gate,
                gate_x,
                use_amp=use_amp,
                use_tta=bool(args.presence_gate_tta),
            )
            gate_prob = float(gate_probs[0].detach().cpu().item())
            gate_keep = gate_prob >= float(presence_gate_threshold)

        probs_sum = None
        for model_idx, (model, weight, size) in enumerate(zip(models, weights, sizes)):
            x = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
            probs = predict_probs(model, x, use_amp=use_amp, use_tta=bool(args.tta))
            probs = F.interpolate(probs.float(), size=(h, w), mode="bilinear", align_corners=False)
            if model_idx in gated_model_indices and not gate_keep:
                probs = torch.zeros_like(probs)
            weighted = float(weight) * probs
            probs_sum = weighted if probs_sum is None else probs_sum + weighted

        pred_mask = (probs_sum[0, 0].detach().cpu().numpy() > float(args.threshold)).astype(np.uint8)

        save_path = args.submission_task_dir / rel.parent / f"{image_path.stem}_label_bin.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").save(save_path)
        records.append(
            {
                "case_id": info["case_id"],
                "segmentation": save_path.relative_to(output_json.parent).as_posix(),
            }
        )
        if gate_prob is None:
            print(f"[predict] {idx}/{len(files)} -> {save_path.name}")
        else:
            print(f"[predict] {idx}/{len(files)} gate={gate_prob:.4f} keep={int(gate_keep)} -> {save_path.name}")

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)
    print(f"Saved json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
