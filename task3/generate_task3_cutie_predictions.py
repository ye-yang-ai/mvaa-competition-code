#!/usr/bin/env python3
"""Generate Task3 predictions with Cutie VOS prior/gate post-processing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CUTIE_ROOT = REPO_ROOT / "external" / "Cutie"
if CUTIE_ROOT.exists():
    sys.path.insert(0, str(CUTIE_ROOT))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from generate_task3_ensemble_predictions import build_model  # noqa: E402
from generate_task3_multi_ensemble_predictions import normalized_weights  # noqa: E402
from generate_task3_predictions import discover_images, pick_device, predict_probs  # noqa: E402

try:
    from cutie.inference.inference_core import InferenceCore  # type: ignore  # noqa: E402
    from cutie.utils.get_default_model import get_default_model  # type: ignore  # noqa: E402
except Exception as exc:  # pragma: no cover
    InferenceCore = None
    get_default_model = None
    _CUTIE_IMPORT_ERROR = exc
else:
    _CUTIE_IMPORT_ERROR = None


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


@dataclass
class FrameItem:
    image_path: Path
    rel_path: Path
    case_id: str
    frame_idx: int
    image_u8: np.ndarray
    prob_ens: np.ndarray
    prob_v15: np.ndarray


@dataclass
class FrameStats:
    area: int
    area_ratio: float
    max_component_area: int
    num_components: int
    mean_fg_prob: float
    high_fg_ratio: float
    centroid: tuple[float, float] | None
    reliable: bool = False
    unstable: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/val/images")
    parser.add_argument("--submission-task-dir", type=Path, required=True)
    parser.add_argument("--video-folders", nargs="*", default=[])
    parser.add_argument("--device", default="auto", help='"auto", "cpu", "cuda", or "cuda:N"')
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")

    parser.add_argument("--base-threshold", type=float, default=0.40)
    parser.add_argument("--v15-gate-threshold", type=float, default=0.285)
    parser.add_argument("--cutie-threshold", type=float, default=0.50)
    parser.add_argument("--final-threshold", type=float, default=0.40)
    parser.add_argument("--anchor-threshold", type=float, default=0.40)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--gap-threshold", type=int, default=8)
    parser.add_argument("--min-anchor-area-ratio-factor", type=float, default=0.35)
    parser.add_argument("--max-anchor-area-ratio-factor", type=float, default=2.50)
    parser.add_argument("--min-anchor-mean-prob", type=float, default=0.55)
    parser.add_argument("--max-anchor-components", type=int, default=5)
    parser.add_argument("--max-area-jump-ratio", type=float, default=0.75)
    parser.add_argument("--max-centroid-jump-frac", type=float, default=0.25)
    parser.add_argument("--cutie-weight", type=float, default=0.35)
    parser.add_argument("--cutie-weight-unstable", type=float, default=0.55)
    parser.add_argument("--cutie-max-internal-size", type=int, default=480)
    parser.add_argument("--disable-backward", action="store_true", default=False)
    parser.add_argument("--gate-dilate-iters", type=int, default=8)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true", default=True)
    parser.add_argument("--no-fill-holes", action="store_false", dest="fill_holes")
    parser.add_argument("--save-debug", action="store_true", default=False)
    return parser.parse_args()


def frame_number(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def connected_component_stats(mask: np.ndarray) -> tuple[int, int]:
    if not bool(mask.any()):
        return 0, 0
    labels, n = ndi.label(mask)
    if n == 0:
        return 0, 0
    counts = np.bincount(labels.ravel())
    max_area = int(counts[1:].max()) if counts.size > 1 else 0
    return int(n), max_area


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def centroid_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def compute_stats(probs: list[np.ndarray], threshold: float) -> list[FrameStats]:
    stats: list[FrameStats] = []
    for prob in probs:
        mask = prob > float(threshold)
        area = int(mask.sum())
        num_components, max_component_area = connected_component_stats(mask)
        mean_fg_prob = float(prob[mask].mean()) if area > 0 else 0.0
        high_fg_ratio = float((prob > max(float(threshold), 0.70)).sum() / max(area, 1))
        stats.append(
            FrameStats(
                area=area,
                area_ratio=float(area / mask.size),
                max_component_area=max_component_area,
                num_components=num_components,
                mean_fg_prob=mean_fg_prob,
                high_fg_ratio=high_fg_ratio,
                centroid=centroid(mask),
            )
        )
    return stats


def mark_reliable_and_unstable(
    stats: list[FrameStats],
    image_shape: tuple[int, int],
    *,
    min_area_factor: float,
    max_area_factor: float,
    min_mean_prob: float,
    max_components: int,
    max_area_jump_ratio: float,
    max_centroid_jump_frac: float,
) -> None:
    positive_areas = [s.area for s in stats if s.area > 0]
    median_area = float(np.median(positive_areas)) if positive_areas else 0.0
    min_area = max(1.0, median_area * float(min_area_factor))
    max_area = max(min_area, median_area * float(max_area_factor)) if median_area > 0 else float("inf")
    diag = float((image_shape[0] ** 2 + image_shape[1] ** 2) ** 0.5)
    max_centroid_jump = float(max_centroid_jump_frac) * diag

    for i, stat in enumerate(stats):
        prev_stat = stats[i - 1] if i > 0 else None
        next_stat = stats[i + 1] if i + 1 < len(stats) else None
        area_jumps = []
        centroid_jumps = []
        for other in (prev_stat, next_stat):
            if other is None or other.area <= 0 or stat.area <= 0:
                continue
            denom = max(float(other.area), float(stat.area), 1.0)
            area_jumps.append(abs(float(stat.area) - float(other.area)) / denom)
            centroid_jumps.append(centroid_distance(stat.centroid, other.centroid))
        area_jump = max(area_jumps) if area_jumps else 0.0
        centroid_jump = max(centroid_jumps) if centroid_jumps else 0.0

        stat.reliable = (
            stat.area >= min_area
            and stat.area <= max_area
            and stat.mean_fg_prob >= float(min_mean_prob)
            and stat.num_components <= int(max_components)
            and area_jump <= float(max_area_jump_ratio)
            and centroid_jump <= max_centroid_jump
        )
        stat.unstable = (
            stat.area == 0
            or stat.num_components > int(max_components)
            or area_jump > float(max_area_jump_ratio)
            or centroid_jump > max_centroid_jump
        )


def split_clips(frames: list[FrameItem], gap_threshold: int) -> list[list[int]]:
    clips: list[list[int]] = []
    current: list[int] = []
    prev_frame = None
    for idx, frame in enumerate(frames):
        if prev_frame is not None and frame.frame_idx - prev_frame > int(gap_threshold):
            if current:
                clips.append(current)
            current = []
        current.append(idx)
        prev_frame = frame.frame_idx
    if current:
        clips.append(current)
    return clips


def choose_anchors(clip_indices: list[int], stats: list[FrameStats], anchor_stride: int) -> list[int]:
    reliable = [idx for idx in clip_indices if stats[idx].reliable]
    if not reliable:
        return []
    anchors = [reliable[0]]
    last_anchor_pos = clip_indices.index(reliable[0])
    for idx in reliable[1:]:
        pos = clip_indices.index(idx)
        if pos - last_anchor_pos >= int(anchor_stride):
            anchors.append(idx)
            last_anchor_pos = pos
    if anchors[-1] != reliable[-1] and len(clip_indices) - last_anchor_pos > int(anchor_stride):
        anchors.append(reliable[-1])
    return sorted(set(anchors))


def image_to_tensor(image_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = image_u8.astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).to(device=device, dtype=torch.float32)


def mask_to_index_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.uint8)).to(device=device, dtype=torch.long)


def cutie_object_prob(output_prob: torch.Tensor) -> np.ndarray:
    if output_prob.ndim != 3 or output_prob.shape[0] < 2:
        return np.zeros(tuple(int(v) for v in output_prob.shape[-2:]), dtype=np.float32)
    return output_prob[1].detach().float().cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_cutie_on_frames(
    frames: list[FrameItem],
    stats: list[FrameStats],
    args: argparse.Namespace,
    device: torch.device,
    cutie,
) -> tuple[np.ndarray, dict]:
    if InferenceCore is None or get_default_model is None:
        raise ImportError(f"Cutie import failed: {_CUTIE_IMPORT_ERROR!r}")
    if device.type != "cuda":
        raise RuntimeError("Cutie default model requires CUDA in this integration.")

    processor_cfg = cutie.cfg

    h, w = frames[0].prob_ens.shape
    cutie_sum = np.zeros((len(frames), h, w), dtype=np.float32)
    cutie_count = np.zeros((len(frames),), dtype=np.float32)
    clips = split_clips(frames, int(args.gap_threshold))
    clip_records = []

    use_backward = not bool(args.disable_backward)
    for clip_id, clip_indices in enumerate(clips, start=1):
        anchors = choose_anchors(clip_indices, stats, int(args.anchor_stride))
        clip_records.append(
            {
                "clip_id": clip_id,
                "frames": [frames[i].image_path.name for i in clip_indices],
                "anchors": [frames[i].image_path.name for i in anchors],
            }
        )
        if not anchors:
            continue

        anchor_positions = [clip_indices.index(a) for a in anchors]

        for anchor_idx, anchor_pos in zip(anchors, anchor_positions):
            end_pos = len(clip_indices)
            later = [p for p in anchor_positions if p > anchor_pos]
            if later:
                end_pos = min(later)

            processor = InferenceCore(cutie, cfg=processor_cfg)
            processor.max_internal_size = int(args.cutie_max_internal_size)
            for pos in range(anchor_pos, end_pos):
                frame_i = clip_indices[pos]
                image_t = image_to_tensor(frames[frame_i].image_u8, device)
                mask_t = None
                objects = None
                if pos == anchor_pos:
                    anchor_mask = frames[frame_i].prob_ens > float(args.anchor_threshold)
                    if not bool(anchor_mask.any()):
                        break
                    mask_t = mask_to_index_tensor(anchor_mask.astype(np.uint8), device)
                    objects = [1]
                with torch.amp.autocast(device_type="cuda", enabled=bool(args.amp)):
                    out = processor.step(image_t, mask_t, objects=objects)
                cutie_sum[frame_i] += cutie_object_prob(out)
                cutie_count[frame_i] += 1.0

        if use_backward:
            for anchor_idx, anchor_pos in zip(anchors, anchor_positions):
                prev_positions = [p for p in anchor_positions if p < anchor_pos]
                start_pos = max(prev_positions) + 1 if prev_positions else 0
                processor = InferenceCore(cutie, cfg=processor_cfg)
                processor.max_internal_size = int(args.cutie_max_internal_size)
                for pos in range(anchor_pos, start_pos - 1, -1):
                    frame_i = clip_indices[pos]
                    image_t = image_to_tensor(frames[frame_i].image_u8, device)
                    mask_t = None
                    objects = None
                    if pos == anchor_pos:
                        anchor_mask = frames[frame_i].prob_ens > float(args.anchor_threshold)
                        if not bool(anchor_mask.any()):
                            break
                        mask_t = mask_to_index_tensor(anchor_mask.astype(np.uint8), device)
                        objects = [1]
                    with torch.amp.autocast(device_type="cuda", enabled=bool(args.amp)):
                        out = processor.step(image_t, mask_t, objects=objects)
                    cutie_sum[frame_i] += cutie_object_prob(out)
                    cutie_count[frame_i] += 1.0

    cutie_probs = np.zeros_like(cutie_sum, dtype=np.float32)
    valid = cutie_count > 0
    cutie_probs[valid] = cutie_sum[valid] / cutie_count[valid, None, None]
    return cutie_probs, {"clips": clip_records, "cutie_count": cutie_count.tolist()}


def postprocess_mask(
    final_prob: np.ndarray,
    cutie_prob: np.ndarray | None,
    v15_prob: np.ndarray,
    *,
    threshold: float,
    cutie_threshold: float,
    v15_gate_threshold: float,
    gate_dilate_iters: int,
    min_component_area: int,
    fill_holes: bool,
) -> np.ndarray:
    pred = final_prob > float(threshold)
    labels, n = ndi.label(pred)
    if n == 0:
        return np.zeros_like(pred, dtype=np.uint8)

    gate = v15_prob > float(v15_gate_threshold)
    if cutie_prob is not None:
        gate = gate | (cutie_prob > float(cutie_threshold))
    if int(gate_dilate_iters) > 0 and bool(gate.any()):
        gate = ndi.binary_dilation(gate, iterations=int(gate_dilate_iters))

    keep = np.zeros_like(pred, dtype=bool)
    counts = np.bincount(labels.ravel())
    for label_id in range(1, n + 1):
        component = labels == label_id
        area = int(counts[label_id]) if label_id < len(counts) else int(component.sum())
        if area < int(min_component_area):
            continue
        if bool(gate.any()) and not bool((component & gate).any()):
            continue
        keep |= component
    if bool(fill_holes) and bool(keep.any()):
        keep = ndi.binary_fill_holes(keep)
    return keep.astype(np.uint8)


@torch.no_grad()
def predict_video_frames(
    models: list[torch.nn.Module],
    sizes: list[tuple[int, int]],
    weights: list[float],
    infos: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> list[FrameItem]:
    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    use_amp = bool(args.amp and device.type == "cuda")

    frames: list[FrameItem] = []
    for idx, info in enumerate(infos, start=1):
        image_path = Path(info["image_path"])
        rel = Path(info["image_rel_path"])
        image_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        h, w = image_u8.shape[:2]
        image_f = image_u8.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
        image_t = (image_t - norm_mean) / norm_std

        probs_sum = None
        prob_v15 = None
        for model_idx, (model, weight, size) in enumerate(zip(models, weights, sizes)):
            x = F.interpolate(image_t, size=size, mode="bilinear", align_corners=False)
            probs = predict_probs(model, x, use_amp=use_amp, use_tta=bool(args.tta))
            probs = F.interpolate(probs.float(), size=(h, w), mode="bilinear", align_corners=False)
            if model_idx == 0:
                prob_v15 = probs[0, 0].detach().cpu().numpy().astype(np.float32)
            weighted = float(weight) * probs
            probs_sum = weighted if probs_sum is None else probs_sum + weighted

        assert probs_sum is not None and prob_v15 is not None
        frames.append(
            FrameItem(
                image_path=image_path,
                rel_path=rel,
                case_id=str(info["case_id"]),
                frame_idx=frame_number(image_path),
                image_u8=image_u8,
                prob_ens=probs_sum[0, 0].detach().cpu().numpy().astype(np.float32),
                prob_v15=prob_v15,
            )
        )
        print(f"[per-frame] {idx}/{len(infos)} {image_path.name}")
    return frames


def group_by_video(files: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for info in files:
        video = Path(info["image_rel_path"]).parent.as_posix()
        grouped.setdefault(video, []).append(info)
    for video in grouped:
        grouped[video].sort(key=lambda x: frame_number(Path(x["image_path"])))
    return dict(sorted(grouped.items()))


def main() -> int:
    args = parse_args()
    if len(args.ckpts) != len(args.weights):
        raise ValueError(f"--ckpts count ({len(args.ckpts)}) must match --weights count ({len(args.weights)}).")
    if not CUTIE_ROOT.exists():
        raise FileNotFoundError(f"Cutie repo not found: {CUTIE_ROOT}")

    output_json = args.submission_task_dir / "task3_predictions.json"
    args.submission_task_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This Cutie integration requires CUDA. Use --device cuda:N.")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    weights = normalized_weights([float(w) for w in args.weights])

    models = []
    sizes = []
    for ckpt_path in args.ckpts:
        model, train_args = build_model(ckpt_path, device=device)
        models.append(model)
        sizes.append(tuple(int(v) for v in train_args.get("image_size", [448, 800])))

    files = discover_images(args.data_dir, IMAGE_EXTS, args.video_folders)
    grouped = group_by_video(files)
    print(f"Device: {device} AMP={args.amp} TTA={args.tta}")
    print(f"Cutie root: {CUTIE_ROOT}")
    print(f"Videos: {len(grouped)} Input images: {len(files)}")
    print(f"Weights: {weights} final_threshold={args.final_threshold}")
    if InferenceCore is None or get_default_model is None:
        raise ImportError(f"Cutie import failed: {_CUTIE_IMPORT_ERROR!r}")
    cutie = get_default_model()

    records: List[Dict[str, str]] = []
    debug = {"videos": {}}
    for video_name, infos in grouped.items():
        print(f"[video] {video_name} frames={len(infos)}")
        frames = predict_video_frames(models, sizes, weights, infos, args, device)
        stats = compute_stats([f.prob_ens for f in frames], float(args.anchor_threshold))
        mark_reliable_and_unstable(
            stats,
            frames[0].prob_ens.shape,
            min_area_factor=float(args.min_anchor_area_ratio_factor),
            max_area_factor=float(args.max_anchor_area_ratio_factor),
            min_mean_prob=float(args.min_anchor_mean_prob),
            max_components=int(args.max_anchor_components),
            max_area_jump_ratio=float(args.max_area_jump_ratio),
            max_centroid_jump_frac=float(args.max_centroid_jump_frac),
        )
        cutie_probs, cutie_meta = run_cutie_on_frames(frames, stats, args, device, cutie)

        video_debug = {
            "stats": [
                {
                    "frame": f.image_path.name,
                    "frame_idx": f.frame_idx,
                    "area": s.area,
                    "components": s.num_components,
                    "mean_fg_prob": s.mean_fg_prob,
                    "reliable": s.reliable,
                    "unstable": s.unstable,
                    "cutie_count": cutie_meta["cutie_count"][i],
                }
                for i, (f, s) in enumerate(zip(frames, stats))
            ],
            "cutie": cutie_meta,
        }
        debug["videos"][video_name] = video_debug

        for i, frame in enumerate(frames):
            has_cutie = cutie_meta["cutie_count"][i] > 0
            cutie_prob = cutie_probs[i] if has_cutie else None
            cutie_w = float(args.cutie_weight_unstable if stats[i].unstable else args.cutie_weight)
            if has_cutie:
                final_prob = (1.0 - cutie_w) * frame.prob_ens + cutie_w * cutie_probs[i]
            else:
                final_prob = frame.prob_ens
            pred_mask = postprocess_mask(
                final_prob,
                cutie_prob,
                frame.prob_v15,
                threshold=float(args.final_threshold),
                cutie_threshold=float(args.cutie_threshold),
                v15_gate_threshold=float(args.v15_gate_threshold),
                gate_dilate_iters=int(args.gate_dilate_iters),
                min_component_area=int(args.min_component_area),
                fill_holes=bool(args.fill_holes),
            )

            save_path = args.submission_task_dir / frame.rel_path.parent / f"{frame.image_path.stem}_label_bin.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").save(save_path)
            records.append(
                {
                    "case_id": frame.case_id,
                    "segmentation": save_path.relative_to(output_json.parent).as_posix(),
                }
            )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)
    print(f"Saved json: {output_json}")
    if bool(args.save_debug):
        debug_path = args.submission_task_dir / "cutie_debug.json"
        with debug_path.open("w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
        print(f"Saved debug: {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
