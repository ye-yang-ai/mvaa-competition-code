#!/usr/bin/env python3
"""Fast weighted Task1 probability ensemble with explicit progress output."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    if len(args.input_dir) != len(args.weight):
        raise ValueError("--input-dir and --weight must have the same count")

    weights = np.asarray(args.weight, dtype=np.float32)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Weights must be non-negative and at least one weight must be positive")
    weights = weights / weights.sum()

    for folder in args.input_dir:
        if not folder.is_dir():
            raise FileNotFoundError(folder)

    first = args.input_dir[0]
    case_names = sorted(p.stem for p in first.glob("*.npz"))
    if not case_names:
        raise RuntimeError(f"No .npz files found in {first}")

    for folder in args.input_dir[1:]:
        missing = [case for case in case_names if not (folder / f"{case}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} missing {len(missing)} cases, first: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Normalized weights:", ", ".join(f"{w:.6f}" for w in weights.tolist()), flush=True)
    for index, case in enumerate(case_names, start=1):
        avg = None
        for folder, weight in zip(args.input_dir, weights):
            probs = np.load(folder / f"{case}.npz")["probabilities"]
            weighted = probs.astype(np.float32, copy=False) * weight
            avg = weighted if avg is None else avg + weighted

        segmentation = np.argmax(avg, axis=0)
        write_segmentation(segmentation, first / f"{case}.pkl", args.output_dir / f"{case}.nii.gz")
        print(f"[{index:02d}/{len(case_names):02d}] wrote {case}.nii.gz", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
