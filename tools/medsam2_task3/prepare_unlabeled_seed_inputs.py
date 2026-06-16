#!/usr/bin/env python3
"""Create MedSAM2 inputs for unlabeled Task3 videos using model-predicted seed masks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK3_DIR = REPO_ROOT / "task3"
if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model_factory import get_model  # noqa: E402


IMG_RE = re.compile(r"^(?P<video>.+)_(?P<idx>\d{6})\.png$")


@dataclass(frozen=True)
class Frame:
    video_id: str
    frame_idx: int
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlabeled-root", type=Path, default=REPO_ROOT / "data/images")
    parser.add_argument("--ckpt-path", type=Path, default=REPO_ROOT / "outputs/exp/task3_unetpp_effb4_none_e100_bs2/checkpoints/best.pt")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/pseudo/task3_medsam2/unlabeled_seed_inputs_debug")
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--exclude-video-ids", nargs="*", default=[])
    parser.add_argument("--max-seeds-per-video", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--window-after", type=int, default=10)
    parser.add_argument("--window-before", type=int, default=0)
    parser.add_argument("--seed-threshold", type=float, default=0.65)
    parser.add_argument("--min-seed-fg-ratio", type=float, default=0.002)
    parser.add_argument("--max-seed-fg-ratio", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    return parser.parse_args()


def discover_frames(video_dir: Path) -> list[Frame]:
    frames: list[Frame] = []
    for image_path in sorted(video_dir.glob("*.png")):
        m = IMG_RE.match(image_path.name)
        if not m:
            continue
        frames.append(Frame(video_id=m.group("video"), frame_idx=int(m.group("idx")), image_path=image_path))
    return frames


def pick_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "efficientnet-b4"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    model = get_model(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    image_size = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    return model, image_size, use_imagenet_norm


@torch.no_grad()
def predict_prob(model, image_path: Path, image_size: tuple[int, int], use_imagenet_norm: bool, device: torch.device, use_amp: bool, use_tta: bool) -> np.ndarray:
    image_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    h, w = image_u8.shape[:2]
    image_f = image_u8.astype(np.float32) / 255.0
    x = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
    x = F.interpolate(x, size=image_size, mode="bilinear", align_corners=False)
    if use_imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
    with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
        prob = torch.sigmoid(model(x))
    if use_tta:
        probs = prob
        for dims in [(3,), (2,), (2, 3)]:
            xf = torch.flip(x, dims=dims)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                pf = torch.sigmoid(model(xf))
            probs = probs + torch.flip(pf, dims=dims)
        prob = probs / 4.0
    prob = F.interpolate(prob.float(), size=(h, w), mode="bilinear", align_corners=False)
    return prob[0, 0].detach().cpu().numpy()


def save_frame_as_jpg(src_png: Path, dst_jpg: Path, jpeg_quality: int) -> None:
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src_png).convert("RGB").save(dst_jpg, quality=jpeg_quality)


def save_mask(mask: np.ndarray, dst_png: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(dst_png)


def main() -> int:
    args = parse_args()
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model, image_size, use_imagenet_norm = load_model(args.ckpt_path, device=device)

    exclude_video_ids = set(str(v) for v in args.exclude_video_ids)
    video_dirs = [p for p in sorted(args.unlabeled_root.iterdir()) if p.is_dir() and p.name not in exclude_video_ids]
    if args.max_videos > 0:
        video_dirs = video_dirs[: args.max_videos]

    clips = []
    seed_rows = []
    skipped = []
    for video_dir in video_dirs:
        frames = discover_frames(video_dir)
        if not frames:
            skipped.append({"video_dir": str(video_dir), "reason": "no_frames"})
            continue
        seeds = []
        for pos, frame in enumerate(frames):
            if pos % max(1, args.frame_stride) != 0:
                continue
            prob = predict_prob(
                model=model,
                image_path=frame.image_path,
                image_size=image_size,
                use_imagenet_norm=use_imagenet_norm,
                device=device,
                use_amp=bool(args.amp),
                use_tta=bool(args.tta),
            )
            mask = (prob >= float(args.seed_threshold)).astype(np.uint8)
            fg_ratio = float(mask.mean())
            if fg_ratio < args.min_seed_fg_ratio or fg_ratio > args.max_seed_fg_ratio:
                seed_rows.append(
                    {
                        "video_id": frame.video_id,
                        "frame_idx": frame.frame_idx,
                        "candidate": 0,
                        "foreground_ratio": fg_ratio,
                        "mean_prob_fg": float(prob[mask > 0].mean()) if mask.any() else 0.0,
                    }
                )
                continue
            seeds.append((pos, mask, fg_ratio, float(prob[mask > 0].mean()) if mask.any() else 0.0))
            seed_rows.append(
                {
                    "video_id": frame.video_id,
                    "frame_idx": frame.frame_idx,
                    "candidate": 1,
                    "foreground_ratio": fg_ratio,
                    "mean_prob_fg": seeds[-1][3],
                }
            )
            if args.max_seeds_per_video > 0 and len(seeds) >= args.max_seeds_per_video:
                break
        if not seeds:
            skipped.append({"video_dir": str(video_dir), "reason": "no_valid_seed"})
            continue
        for seed_pos, seed_mask, fg_ratio, mean_prob_fg in seeds:
            seed = frames[seed_pos]
            start = max(0, seed_pos - args.window_before)
            end = min(len(frames), seed_pos + args.window_after + 1)
            clip = frames[start:end]
            seed_order = seed_pos - start
            clip_name = f"{seed.video_id}__modelseed_{seed.frame_idx:06d}"
            frames_dir = args.output_root / "videos" / clip_name
            masks_dir = args.output_root / "masks" / clip_name
            for order, frame in enumerate(clip):
                save_frame_as_jpg(frame.image_path, frames_dir / f"{order:05d}.jpg", jpeg_quality=args.jpeg_quality)
            save_mask(seed_mask, masks_dir / f"{seed_order:05d}.png")
            clips.append(
                {
                    "clip_name": clip_name,
                    "source_video_id": seed.video_id,
                    "seed_source_frame_idx": seed.frame_idx,
                    "seed_order": seed_order,
                    "seed_mask_stats": {
                        "foreground_pixels": float(seed_mask.sum()),
                        "foreground_ratio": fg_ratio,
                        "mean_prob_fg": mean_prob_fg,
                        "seed_threshold": float(args.seed_threshold),
                    },
                    "frames": [
                        {
                            "order": order,
                            "frame_name": f"{order:05d}",
                            "source_video_id": frame.video_id,
                            "source_frame_idx": frame.frame_idx,
                            "image_path": str(frame.image_path),
                        }
                        for order, frame in enumerate(clip)
                    ],
                }
            )

    video_list_path = args.output_root / "video_list.txt"
    video_list_path.write_text("\n".join(c["clip_name"] for c in clips) + ("\n" if clips else ""), encoding="utf-8")
    manifest = {
        "unlabeled_root": str(args.unlabeled_root),
        "ckpt_path": str(args.ckpt_path),
        "output_root": str(args.output_root),
        "video_list_file": str(video_list_path),
        "exclude_video_ids": sorted(exclude_video_ids),
        "num_clips": len(clips),
        "clips": clips,
        "skipped": skipped,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    seed_csv = args.output_root / "seed_candidates.csv"
    with seed_csv.open("w", encoding="utf-8", newline="") as f:
        if seed_rows:
            writer = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
    print(json.dumps({"manifest": str(manifest_path), "seed_candidates": str(seed_csv), "num_clips": len(clips)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
