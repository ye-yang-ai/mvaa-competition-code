#!/usr/bin/env python3
"""Prepare a Task1 nnU-Net dataset with selected pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference_data/t1_ct"))
    parser.add_argument("--pseudo-dir", type=Path, default=Path("outputs/pseudo/task1_v17_high759"))
    parser.add_argument(
        "--selected-cases",
        type=Path,
        default=Path("outputs/analysis/task1_pseudo/pseudo_top100_cases.txt"),
        help="Text file containing original unlabeled case ids, one per line.",
    )
    parser.add_argument("--nnunet-root", type=Path, default=Path("outputs/nnunet_task1_pseudo_top100"))
    parser.add_argument("--dataset-id", type=int, default=111)
    parser.add_argument("--dataset-name", type=str, default="MVAA_Task1_PseudoTop100")
    parser.add_argument(
        "--source-splits-json",
        type=Path,
        default=Path("outputs/nnunet_task1_fold0/nnUNet_preprocessed/Dataset101_MVAA_Task1/splits_final.json"),
    )
    parser.add_argument("--link-mode", type=str, default="symlink", choices=("symlink", "hardlink", "copy"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--splits-only", action="store_true")
    return parser.parse_args()


def link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def validate_binary_label(path: Path) -> None:
    data = np.asanyarray(nib.load(str(path)).dataobj)
    values = np.unique(data)
    bad = [float(v) for v in values.tolist() if v not in (0, 1)]
    if bad:
        raise ValueError(f"Label {path} has non-binary values: {bad[:10]}")


def read_selected_cases(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected case list: {path}")
    cases = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise RuntimeError(f"No selected cases in {path}")
    return cases


def write_pseudo_splits(args: argparse.Namespace, dataset_dir: Path, pseudo_case_ids: list[str]) -> None:
    source_splits = json.loads(args.source_splits_json.read_text(encoding="utf-8"))
    out_splits = []
    for fold in source_splits:
        train = list(fold["train"]) + pseudo_case_ids
        val = list(fold["val"])
        out_splits.append({"train": train, "val": val})

    preprocessed_dataset_dir = (
        args.nnunet_root / "nnUNet_preprocessed" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    )
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)
    (preprocessed_dataset_dir / "splits_final.json").write_text(json.dumps(out_splits, indent=2), encoding="utf-8")

    split_record = {
        "source_splits_json": args.source_splits_json.as_posix(),
        "dataset_dir": dataset_dir.as_posix(),
        "pseudo_cases": pseudo_case_ids,
        "folds": [
            {
                "fold": idx,
                "train_count": len(fold["train"]),
                "val_count": len(fold["val"]),
                "pseudo_train_count": len(pseudo_case_ids),
                "val_cases": fold["val"],
            }
            for idx, fold in enumerate(out_splits)
        ],
    }
    (args.nnunet_root / "pseudo_splits_record.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw_base = args.nnunet_root / "nnUNet_raw"
    dataset_dir = raw_base / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    selected_cases = read_selected_cases(args.selected_cases)
    pseudo_case_ids = [f"pseudo_{case_id}" for case_id in selected_cases]

    if args.splits_only:
        write_pseudo_splits(args, dataset_dir, pseudo_case_ids)
        print(f"Wrote pseudo splits for {len(pseudo_case_ids)} pseudo cases")
        return 0

    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_dir} exists. Use --overwrite to rebuild it.")
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"

    train_mapping = []
    real_images = sorted((args.data_root / "train" / "labeled" / "images").glob("*.nii.gz"))
    for image_path in real_images:
        case_id = image_path.name.removesuffix(".nii.gz")
        label_path = args.data_root / "train" / "labeled" / "labels" / f"{case_id}-seg.nii.gz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing real label for {case_id}: {label_path}")
        validate_binary_label(label_path)
        link_file(image_path, images_tr / f"{case_id}_0000.nii.gz", args.link_mode)
        link_file(label_path, labels_tr / f"{case_id}.nii.gz", args.link_mode)
        train_mapping.append(
            {
                "case_id": case_id,
                "source": "real",
                "image": image_path.as_posix(),
                "label": label_path.as_posix(),
            }
        )

    manifest = json.loads((args.pseudo_dir / "manifest.json").read_text(encoding="utf-8"))["cases"]
    manifest_by_case = {item["case_id"]: item for item in manifest}
    pseudo_mapping = []
    for case_id in selected_cases:
        item = manifest_by_case.get(case_id)
        if item is None:
            raise KeyError(f"Selected case {case_id} not found in pseudo manifest")
        nnunet_case_id = item["nnunet_case_id"]
        image_path = Path(item["source_image"])
        label_path = args.pseudo_dir / "pred" / f"{nnunet_case_id}.nii.gz"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing pseudo source image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing pseudo label: {label_path}")
        validate_binary_label(label_path)
        link_file(image_path, images_tr / f"{nnunet_case_id}_0000.nii.gz", args.link_mode)
        link_file(label_path, labels_tr / f"{nnunet_case_id}.nii.gz", args.link_mode)
        pseudo_mapping.append(
            {
                "case_id": nnunet_case_id,
                "original_case_id": case_id,
                "source": "pseudo",
                "image": image_path.as_posix(),
                "label": label_path.as_posix(),
            }
        )

    test_mapping = []
    for image_path in sorted((args.data_root / "val" / "images").glob("*.nii.gz")):
        case_id = image_path.name.removesuffix(".nii.gz")
        link_file(image_path, images_ts / f"{case_id}_0000.nii.gz", args.link_mode)
        test_mapping.append({"case_id": case_id, "image": image_path.as_posix()})

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "target": 1},
        "numTraining": len(train_mapping) + len(pseudo_mapping),
        "file_ending": ".nii.gz",
    }
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    mapping = {
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "dataset_dir": dataset_dir.as_posix(),
        "real_train_count": len(train_mapping),
        "pseudo_train_count": len(pseudo_mapping),
        "train": train_mapping + pseudo_mapping,
        "pseudo_train": pseudo_mapping,
        "test": test_mapping,
    }
    args.nnunet_root.mkdir(parents=True, exist_ok=True)
    (args.nnunet_root / "task1_case_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    write_pseudo_splits(args, dataset_dir, pseudo_case_ids)

    print(f"Prepared dataset: {dataset_dir}")
    print(f"Real train: {len(train_mapping)}")
    print(f"Pseudo train: {len(pseudo_mapping)}")
    print(f"Test: {len(test_mapping)}")
    print(f"Mapping: {args.nnunet_root / 'task1_case_mapping.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
