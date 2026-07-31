#!/usr/bin/env python3
"""Weighted ensemble for Task1 nnU-Net probability predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if len(args.input_dir) != len(args.weight):
        raise ValueError("--input-dir and --weight must have the same count")

    weights = np.asarray(args.weight, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    if float(weights.sum()) <= 0:
        raise ValueError("At least one weight must be positive")
    weights = weights / weights.sum()

    input_dirs = args.input_dir
    for folder in input_dirs:
        if not folder.is_dir():
            raise FileNotFoundError(folder)

    first = input_dirs[0]
    dataset_json = load_json(first / "dataset.json")
    plans = load_json(first / "plans.json")
    plans_manager = PlansManager(plans)
    label_manager: LabelManager = plans_manager.get_label_manager(dataset_json)
    writer = plans_manager.image_reader_writer_class()
    file_ending = dataset_json["file_ending"]

    case_names = sorted(p.stem for p in first.glob("*.npz"))
    if not case_names:
        raise RuntimeError(f"No .npz files found in {first}")

    for folder in input_dirs[1:]:
        missing = [case for case in case_names if not (folder / f"{case}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} missing {len(missing)} cases, first: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first / "dataset.json", args.output_dir / "dataset.json")
    shutil.copy2(first / "plans.json", args.output_dir / "plans.json")

    for case in case_names:
        avg = None
        for folder, weight in zip(input_dirs, weights):
            probs = np.load(folder / f"{case}.npz")["probabilities"]
            if avg is None:
                avg = probs.astype(np.float32, copy=True) * weight
            else:
                if probs.shape != avg.shape:
                    raise ValueError(f"Shape mismatch for {case}: {probs.shape} vs {avg.shape}")
                avg += probs.astype(np.float32, copy=False) * weight

        segmentation = label_manager.convert_probabilities_to_segmentation(avg)
        properties = load_pickle(str(first / f"{case}.pkl"))
        writer.write_seg(segmentation, str(args.output_dir / f"{case}{file_ending}"), properties)
        if args.save_probabilities:
            np.savez_compressed(args.output_dir / f"{case}.npz", probabilities=avg)
            shutil.copy2(first / f"{case}.pkl", args.output_dir / f"{case}.pkl")

    print(f"Ensembled {len(case_names)} cases into {args.output_dir}")
    print("Weights:", ", ".join(f"{w:.6f}" for w in weights.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
