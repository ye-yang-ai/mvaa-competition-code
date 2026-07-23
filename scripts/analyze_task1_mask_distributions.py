#!/usr/bin/env python3
"""Analyze Task1 mask/image distributions for pseudo-label self-training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference_data/t1_ct"))
    parser.add_argument(
        "--v17-task1-dir",
        type=Path,
        default=Path("outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission/t1_ct"),
    )
    parser.add_argument(
        "--v16-task1-dir",
        type=Path,
        default=Path("outputs/submissions/submit_v16_task1_nnunet5fold_task2_v14_task3_thr0285/submission/t1_ct"),
    )
    parser.add_argument(
        "--v18-task1-dir",
        type=Path,
        default=Path("outputs/submissions/submit_v18_task1_v17_post_min100_task2_v14_task3_v15/submission/t1_ct"),
    )
    parser.add_argument(
        "--pseudo-task1-dir",
        type=Path,
        default=None,
        help="Optional pseudo-label prediction dir with task1_predictions.json or *-pred.nii.gz files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/task1_pseudo"))
    return parser.parse_args()


def spacing_of(img: nib.Nifti1Image) -> tuple[float, float, float]:
    zooms = img.header.get_zooms()[:3]
    return tuple(float(x) for x in zooms)


def bbox_sizes(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return (0, 0, 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple(int(x) for x in (maxs - mins + 1))


def mask_record(path: Path, source: str, case_id: str, image_path: Path | None = None) -> dict:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = data > 0
    spacing = spacing_of(img)
    voxel_volume = float(np.prod(spacing))
    fg_voxels = int(mask.sum())
    image_voxels = int(np.prod(mask.shape))

    structure = ndi.generate_binary_structure(mask.ndim, 1)
    labeled, n_components = ndi.label(mask, structure=structure)
    if n_components > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        component_sizes = sorted((int(x) for x in sizes if x > 0), reverse=True)
    else:
        component_sizes = []
    largest = component_sizes[0] if component_sizes else 0
    second = component_sizes[1] if len(component_sizes) > 1 else 0
    small_lt_10 = sum(1 for x in component_sizes if x < 10)
    small_lt_50 = sum(1 for x in component_sizes if x < 50)
    small_lt_100 = sum(1 for x in component_sizes if x < 100)

    image_shape = None
    image_spacing = None
    if image_path is not None and image_path.exists():
        image = nib.load(str(image_path))
        image_shape = tuple(int(x) for x in image.shape[:3])
        image_spacing = spacing_of(image)

    return {
        "source": source,
        "case_id": case_id,
        "path": path.as_posix(),
        "image_path": image_path.as_posix() if image_path is not None else "",
        "shape_x": int(mask.shape[0]),
        "shape_y": int(mask.shape[1]),
        "shape_z": int(mask.shape[2]),
        "spacing_x": spacing[0],
        "spacing_y": spacing[1],
        "spacing_z": spacing[2],
        "image_shape": str(image_shape) if image_shape is not None else "",
        "image_spacing": str(image_spacing) if image_spacing is not None else "",
        "fg_voxels": fg_voxels,
        "image_voxels": image_voxels,
        "fg_ratio": fg_voxels / image_voxels if image_voxels > 0 else 0.0,
        "physical_volume_mm3": fg_voxels * voxel_volume,
        "components": int(n_components),
        "largest_component": largest,
        "second_component": second,
        "largest_component_ratio": largest / fg_voxels if fg_voxels > 0 else 0.0,
        "small_components_lt10": int(small_lt_10),
        "small_components_lt50": int(small_lt_50),
        "small_components_lt100": int(small_lt_100),
        "bbox_x": bbox_sizes(mask)[0],
        "bbox_y": bbox_sizes(mask)[1],
        "bbox_z": bbox_sizes(mask)[2],
    }


def summarize_values(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {}
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(q1),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p75": float(q3),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "iqr": float(iqr),
        "tukey_low": float(q1 - 1.5 * iqr),
        "tukey_high": float(q3 + 1.5 * iqr),
    }


def task1_cases(task_dir: Path) -> list[tuple[str, Path]]:
    pred_json = task_dir / "task1_predictions.json"
    if pred_json.exists():
        data = json.loads(pred_json.read_text(encoding="utf-8"))
        return [(item["case_id"], task_dir / item["segmentation"]) for item in data["cases"]]
    return [(p.name.removesuffix("-pred.nii.gz"), p) for p in sorted(task_dir.glob("*-pred.nii.gz"))]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(summary: dict, comparisons: list[dict], image_summary: dict) -> str:
    lines = [
        "# Task1 pseudo-label distribution analysis",
        "",
        "## Mask distributions",
        "",
        "| Source | Count | Volume median | Volume p05-p95 | FG vox median | Components median | Largest ratio median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, stats in summary.items():
        vol = stats["physical_volume_mm3"]
        vox = stats["fg_voxels"]
        comp = stats["components"]
        ratio = stats["largest_component_ratio"]
        lines.append(
            f"| {source} | {vol['count']} | {vol['median']:.1f} | {vol['p05']:.1f}-{vol['p95']:.1f} | "
            f"{vox['median']:.1f} | {comp['median']:.1f} | {ratio['median']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Prediction deltas against v17",
            "",
            "| Source | Cases | Total delta voxels | Median abs delta | Max abs delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['source']} | {item['cases']} | {item['total_delta_voxels']} | "
            f"{item['median_abs_delta_voxels']:.1f} | {item['max_abs_delta_voxels']} |"
        )

    lines.extend(
        [
            "",
            "## Unlabeled image metadata",
            "",
            "| Field | Count | Median | p05-p95 | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for field, stats in image_summary.items():
        lines.append(
            f"| {field} | {stats['count']} | {stats['median']:.3f} | "
            f"{stats['p05']:.3f}-{stats['p95']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def image_records(paths: list[Path], source: str) -> list[dict]:
    rows = []
    for path in paths:
        img = nib.load(str(path))
        spacing = spacing_of(img)
        shape = tuple(int(x) for x in img.shape[:3])
        rows.append(
            {
                "source": source,
                "case_id": path.name.removesuffix(".nii.gz"),
                "path": path.as_posix(),
                "shape_x": shape[0],
                "shape_y": shape[1],
                "shape_z": shape[2],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "image_voxels": int(np.prod(shape)),
                "physical_volume_mm3": float(np.prod(shape) * np.prod(spacing)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    label_dir = args.data_root / "train" / "labeled" / "labels"
    image_dir = args.data_root / "train" / "labeled" / "images"
    val_image_dir = args.data_root / "val" / "images"
    for label_path in sorted(label_dir.glob("*-seg.nii.gz")):
        case_id = label_path.name.removesuffix("-seg.nii.gz")
        rows.append(mask_record(label_path, "labeled_gt", case_id, image_dir / f"{case_id}.nii.gz"))

    for source, task_dir in (
        ("v17_val_pred", args.v17_task1_dir),
        ("v16_val_pred", args.v16_task1_dir),
        ("v18_v17_min100_pred", args.v18_task1_dir),
    ):
        if task_dir is None or not task_dir.exists():
            continue
        for case_id, pred_path in task1_cases(task_dir):
            rows.append(mask_record(pred_path, source, case_id, val_image_dir / f"{case_id}.nii.gz"))

    if args.pseudo_task1_dir is not None:
        for case_id, pred_path in task1_cases(args.pseudo_task1_dir):
            rows.append(mask_record(pred_path, "pseudo_unlabeled_pred", case_id, None))

    write_csv(out_dir / "mask_stats.csv", rows)

    summary = {}
    for source in sorted({row["source"] for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        summary[source] = {
            "fg_voxels": summarize_values(row["fg_voxels"] for row in subset),
            "fg_ratio": summarize_values(row["fg_ratio"] for row in subset),
            "physical_volume_mm3": summarize_values(row["physical_volume_mm3"] for row in subset),
            "components": summarize_values(row["components"] for row in subset),
            "largest_component_ratio": summarize_values(row["largest_component_ratio"] for row in subset),
            "second_component": summarize_values(row["second_component"] for row in subset),
            "bbox_x": summarize_values(row["bbox_x"] for row in subset),
            "bbox_y": summarize_values(row["bbox_y"] for row in subset),
            "bbox_z": summarize_values(row["bbox_z"] for row in subset),
        }

    by_source_case = {(row["source"], row["case_id"]): row for row in rows}
    comparisons = []
    v17_cases = sorted(row["case_id"] for row in rows if row["source"] == "v17_val_pred")
    for source in ("v16_val_pred", "v18_v17_min100_pred"):
        deltas = []
        for case_id in v17_cases:
            base = by_source_case.get(("v17_val_pred", case_id))
            other = by_source_case.get((source, case_id))
            if base is None or other is None:
                continue
            deltas.append(other["fg_voxels"] - base["fg_voxels"])
        if deltas:
            abs_deltas = [abs(x) for x in deltas]
            comparisons.append(
                {
                    "source": source,
                    "cases": len(deltas),
                    "total_delta_voxels": int(sum(deltas)),
                    "median_abs_delta_voxels": float(np.median(abs_deltas)),
                    "max_abs_delta_voxels": int(max(abs_deltas)),
                }
            )

    unlabeled_paths = sorted((args.data_root / "train" / "unlabeled").glob("*.nii.gz"))
    unlabeled_rows = image_records(unlabeled_paths, "unlabeled_image")
    write_csv(out_dir / "unlabeled_image_stats.csv", unlabeled_rows)
    image_summary = {
        field: summarize_values(row[field] for row in unlabeled_rows)
        for field in ("shape_x", "shape_y", "shape_z", "spacing_x", "spacing_y", "spacing_z", "image_voxels", "physical_volume_mm3")
    }

    result = {
        "summary": summary,
        "comparisons_against_v17": comparisons,
        "unlabeled_image_summary": image_summary,
        "outputs": {
            "mask_stats_csv": (out_dir / "mask_stats.csv").as_posix(),
            "unlabeled_image_stats_csv": (out_dir / "unlabeled_image_stats.csv").as_posix(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(markdown_summary(summary, comparisons, image_summary), encoding="utf-8")

    print(f"Mask rows: {len(rows)}")
    print(f"Unlabeled images: {len(unlabeled_rows)}")
    print(f"Wrote: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
