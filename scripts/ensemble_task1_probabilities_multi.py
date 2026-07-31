#!/usr/bin/env python3
"""Generate multiple weighted Task1 probability ensembles in one pass."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_candidate(value: str) -> tuple[str, np.ndarray]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME:w1,w2,w3")
    name, raw_weights = value.split(":", 1)
    weights = np.asarray([float(x) for x in raw_weights.split(",")], dtype=np.float32)
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise argparse.ArgumentTypeError("candidate weights must be non-negative and non-zero")
    return name, weights / weights.sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    return parser.parse_args()


def write_segmentation(segmentation: np.ndarray, properties_path: Path, output_path: Path) -> None:
    with properties_path.open("rb") as f:
        properties = pickle.load(f)

    image = sitk.GetImageFromArray(segmentation.astype(np.uint8, copy=False))
    sitk_stuff = properties["sitk_stuff"]
    image.SetSpacing(tuple(float(x) for x in sitk_stuff["spacing"]))
    image.SetOrigin(tuple(float(x) for x in sitk_stuff["origin"]))
    image.SetDirection(tuple(float(x) for x in sitk_stuff["direction"]))
    sitk.WriteImage(image, str(output_path), True)


def main() -> int:
    args = parse_args()
    input_dirs = args.input_dir
    for folder in input_dirs:
        if not folder.is_dir():
            raise FileNotFoundError(folder)

    candidates = args.candidate
    for name, weights in candidates:
        if len(weights) != len(input_dirs):
            raise ValueError(f"{name}: expected {len(input_dirs)} weights, got {len(weights)}")
        (args.output_root / name / "nnunet_task1_pred").mkdir(parents=True, exist_ok=True)

    first = input_dirs[0]
    case_names = sorted(p.stem for p in first.glob("*.npz"))
    if not case_names:
        raise RuntimeError(f"No .npz files found in {first}")

    for folder in input_dirs[1:]:
        missing = [case for case in case_names if not (folder / f"{case}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} missing {len(missing)} cases, first: {missing[:5]}")

    for name, weights in candidates:
        print(f"{name}: " + ", ".join(f"{w:.6f}" for w in weights.tolist()), flush=True)

    for index, case in enumerate(case_names, start=1):
        probs = [
            np.load(folder / f"{case}.npz")["probabilities"].astype(np.float32, copy=False)
            for folder in input_dirs
        ]
        for name, weights in candidates:
            avg = np.zeros_like(probs[0], dtype=np.float32)
            for prob, weight in zip(probs, weights):
                avg += prob * weight
            segmentation = np.argmax(avg, axis=0)
            write_segmentation(
                segmentation,
                first / f"{case}.pkl",
                args.output_root / name / "nnunet_task1_pred" / f"{case}.nii.gz",
            )
        print(f"[{index:02d}/{len(case_names):02d}] wrote {case} for {len(candidates)} candidates", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
