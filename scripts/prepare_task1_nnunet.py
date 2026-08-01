#!/usr/bin/env python3
"""Prepare MVAA Task1 data as an nnU-Net v2 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference_data/t1_ct"))
    parser.add_argument("--nnunet-root", type=Path, default=Path("outputs/nnunet_task1_fold0"))
    parser.add_argument("--dataset-id", type=int, default=101)
    parser.add_argument("--dataset-name", type=str, default="MVAA_Task1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_nifti(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def validate_binary_label(path: Path) -> None:
    data = np.asanyarray(nib.load(str(path)).dataobj)
    values = np.unique(data)
    bad = [float(v) for v in values.tolist() if v not in (0, 1)]
    if bad:
        raise ValueError(f"Label {path} has non-binary values: {bad[:10]}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    raw_base = args.nnunet_root / "nnUNet_raw"
    dataset_dir = raw_base / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_dir} exists. Use --overwrite to rebuild it.")
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"

    train_images = sorted((data_root / "train" / "labeled" / "images").glob("*.nii.gz"))
    val_images = sorted((data_root / "val" / "images").glob("*.nii.gz"))
    if not train_images:
        raise FileNotFoundError("No training images found.")
    if not val_images:
        raise FileNotFoundError("No validation/test images found.")

    train_mapping = []
    for image_path in train_images:
        case_id = image_path.name.removesuffix(".nii.gz")
        label_path = data_root / "train" / "labeled" / "labels" / f"{case_id}-seg.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")
        validate_binary_label(label_path)
        copy_nifti(image_path, images_tr / f"{case_id}_0000.nii.gz")
        copy_nifti(label_path, labels_tr / f"{case_id}.nii.gz")
        train_mapping.append({"case_id": case_id, "image": image_path.as_posix(), "label": label_path.as_posix()})

    test_mapping = []
    for image_path in val_images:
        case_id = image_path.name.removesuffix(".nii.gz")
        copy_nifti(image_path, images_ts / f"{case_id}_0000.nii.gz")
        test_mapping.append({"case_id": case_id, "image": image_path.as_posix()})

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "target": 1},
        "numTraining": len(train_mapping),
        "file_ending": ".nii.gz",
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    mapping = {
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "dataset_dir": dataset_dir.as_posix(),
        "train": train_mapping,
        "test": test_mapping,
    }
    args.nnunet_root.mkdir(parents=True, exist_ok=True)
    (args.nnunet_root / "task1_case_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    print(f"Prepared {len(train_mapping)} training cases and {len(test_mapping)} test cases.")
    print(f"Dataset: {dataset_dir}")
    print(f"Mapping: {args.nnunet_root / 'task1_case_mapping.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
