#!/usr/bin/env python3
"""Generate Task1 post-processing submission candidates from existing masks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


TASK_DIRS = ("t1_ct", "t2_tee", "t3_vid")


@dataclass(frozen=True)
class Variant:
    name: str
    source: str
    min_size: int = 0
    keep_components: int = 0
    fill_holes: bool = False
    close_iters: int = 0
    mode: str = "postprocess"


DEFAULT_VARIANTS = (
    Variant("v17_min10", "v17", min_size=10),
    Variant("v17_min20", "v17", min_size=20),
    Variant("v17_min50", "v17", min_size=50),
    Variant("v17_min100", "v17", min_size=100),
    Variant("v17_keep1_min0", "v17", keep_components=1),
    Variant("v17_keep1_min20", "v17", min_size=20, keep_components=1),
    Variant("v17_fillholes", "v17", fill_holes=True),
    Variant("v16_min10", "v16", min_size=10),
    Variant("v16_min20", "v16", min_size=20),
    Variant("v16_min50", "v16", min_size=50),
    Variant("v16_min100", "v16", min_size=100),
    Variant("v16_keep1_min20", "v16", min_size=20, keep_components=1),
    Variant("v16_v17_intersection", "both", mode="intersection"),
    Variant("v16_v17_union", "both", mode="union"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-submission-dir",
        type=Path,
        default=Path("outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission"),
        help="Submission directory used as the base for unchanged tasks.",
    )
    parser.add_argument(
        "--v17-task1-dir",
        type=Path,
        default=Path("outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission/t1_ct"),
        help="Task1 v17/checkpoint_best submission task directory.",
    )
    parser.add_argument(
        "--v16-task1-dir",
        type=Path,
        default=Path("outputs/submissions/submit_v16_task1_nnunet5fold_task2_v14_task3_thr0285/submission/t1_ct"),
        help="Task1 v16/checkpoint_final submission task directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/submissions/task1_postprocess_grid"),
        help="Directory where candidate submissions are written.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-zip", action="store_true", help="Do not create submission.zip for each candidate.")
    return parser.parse_args()


def load_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return img, (data > 0).astype(np.uint8)


def save_mask(mask: np.ndarray, ref_img: nib.Nifti1Image, path: Path) -> None:
    out = nib.Nifti1Image(mask.astype(np.uint8), affine=ref_img.affine, header=ref_img.header.copy())
    nib.save(out, str(path))


def component_stats(mask: np.ndarray) -> tuple[int, list[int]]:
    labeled, n_labels = ndi.label(mask, structure=ndi.generate_binary_structure(mask.ndim, 1))
    if n_labels == 0:
        return 0, []
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return n_labels, sorted((int(x) for x in sizes if x > 0), reverse=True)


def postprocess_mask(mask: np.ndarray, variant: Variant) -> np.ndarray:
    binary = mask.astype(bool)
    structure = ndi.generate_binary_structure(binary.ndim, 1)

    if variant.close_iters > 0:
        binary = ndi.binary_closing(binary, structure=structure, iterations=variant.close_iters)
    if variant.fill_holes:
        binary = ndi.binary_fill_holes(binary)

    labeled, n_labels = ndi.label(binary, structure=structure)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    component_ids = np.argsort(sizes)[::-1]

    kept = np.zeros_like(binary, dtype=bool)
    kept_count = 0
    for component_id in component_ids:
        size = int(sizes[component_id])
        if component_id == 0 or size <= 0:
            continue
        if size < variant.min_size:
            continue
        kept |= labeled == component_id
        kept_count += 1
        if variant.keep_components > 0 and kept_count >= variant.keep_components:
            break

    return kept.astype(np.uint8)


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_base_submission(base_dir: Path, out_submission_dir: Path) -> None:
    for task in TASK_DIRS:
        if task == "t1_ct":
            continue
        src = base_dir / task
        dst = out_submission_dir / task
        if not src.is_dir():
            raise FileNotFoundError(f"Missing base task directory: {src}")
        shutil.copytree(src, dst)


def package_submission(submission_dir: Path, output_zip: Path) -> int:
    written = 0
    with ZipFile(output_zip, "w", ZIP_DEFLATED) as zf:
        for task in TASK_DIRS:
            task_dir = submission_dir / task
            if not task_dir.is_dir():
                raise FileNotFoundError(f"Missing task directory: {task_dir}")
            for path in sorted(task_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(submission_dir).as_posix())
                    written += 1
    return written


def read_cases(task1_dir: Path) -> list[dict]:
    pred_json = task1_dir / "task1_predictions.json"
    data = json.loads(pred_json.read_text(encoding="utf-8"))
    return data["cases"]


def build_variant(variant: Variant, args: argparse.Namespace) -> dict:
    variant_root = args.output_root / variant.name
    submission_dir = variant_root / "submission"
    task1_out_dir = submission_dir / "t1_ct"
    ensure_clean_dir(variant_root, args.overwrite)
    submission_dir.mkdir(parents=True, exist_ok=True)
    task1_out_dir.mkdir(parents=True, exist_ok=True)
    copy_base_submission(args.base_submission_dir, submission_dir)

    cases = read_cases(args.v17_task1_dir)
    records = []
    case_reports = []

    for item in cases:
        case_id = item["case_id"]
        filename = item["segmentation"]
        v17_path = args.v17_task1_dir / filename
        v16_path = args.v16_task1_dir / filename
        ref_img, v17_mask = load_mask(v17_path)

        if variant.mode == "intersection":
            _, v16_mask = load_mask(v16_path)
            out_mask = ((v16_mask > 0) & (v17_mask > 0)).astype(np.uint8)
        elif variant.mode == "union":
            _, v16_mask = load_mask(v16_path)
            out_mask = ((v16_mask > 0) | (v17_mask > 0)).astype(np.uint8)
        else:
            if variant.source == "v16":
                ref_img, src_mask = load_mask(v16_path)
            else:
                src_mask = v17_mask
            out_mask = postprocess_mask(src_mask, variant)

        before_components, before_sizes = component_stats(v17_mask)
        after_components, after_sizes = component_stats(out_mask)
        before_voxels = int(v17_mask.sum())
        after_voxels = int(out_mask.sum())

        save_path = task1_out_dir / filename
        save_mask(out_mask, ref_img, save_path)
        records.append({"case_id": case_id, "segmentation": filename})
        case_reports.append(
            {
                "case_id": case_id,
                "source": variant.source,
                "mode": variant.mode,
                "voxels_v17_before": before_voxels,
                "voxels_after": after_voxels,
                "delta_voxels_after_minus_v17": after_voxels - before_voxels,
                "components_v17_before": before_components,
                "components_after": after_components,
                "largest_components_v17_before": before_sizes[:5],
                "largest_components_after": after_sizes[:5],
            }
        )

    (task1_out_dir / "task1_predictions.json").write_text(
        json.dumps({"cases": records}, indent=2), encoding="utf-8"
    )

    report = {
        "variant": variant.__dict__,
        "base_submission_dir": args.base_submission_dir.as_posix(),
        "v17_task1_dir": args.v17_task1_dir.as_posix(),
        "v16_task1_dir": args.v16_task1_dir.as_posix(),
        "submission_dir": submission_dir.as_posix(),
        "cases": case_reports,
        "total_delta_voxels_after_minus_v17": int(sum(x["delta_voxels_after_minus_v17"] for x in case_reports)),
        "total_abs_delta_voxels_after_minus_v17": int(
            sum(abs(x["delta_voxels_after_minus_v17"]) for x in case_reports)
        ),
    }
    (variant_root / "task1_postprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    zip_files = None
    if not args.no_zip:
        zip_files = package_submission(submission_dir, variant_root / "submission.zip")

    return {
        "variant": variant.name,
        "submission_dir": submission_dir.as_posix(),
        "submission_zip": (variant_root / "submission.zip").as_posix() if not args.no_zip else "",
        "zip_files": zip_files,
        "total_delta_voxels_after_minus_v17": report["total_delta_voxels_after_minus_v17"],
        "total_abs_delta_voxels_after_minus_v17": report["total_abs_delta_voxels_after_minus_v17"],
    }


def main() -> int:
    args = parse_args()
    for task in TASK_DIRS:
        if not (args.base_submission_dir / task).is_dir():
            raise FileNotFoundError(f"Missing base submission task directory: {args.base_submission_dir / task}")
    if not (args.v17_task1_dir / "task1_predictions.json").is_file():
        raise FileNotFoundError(f"Missing v17 task1_predictions.json: {args.v17_task1_dir}")
    if not (args.v16_task1_dir / "task1_predictions.json").is_file():
        raise FileNotFoundError(f"Missing v16 task1_predictions.json: {args.v16_task1_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [build_variant(variant, args) for variant in DEFAULT_VARIANTS]

    summary_json = args.output_root / "summary.json"
    summary_json.write_text(json.dumps({"variants": rows}, indent=2), encoding="utf-8")

    summary_csv = args.output_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} variants to {args.output_root}")
    print(f"Summary: {summary_csv}")
    for row in rows:
        print(
            row["variant"],
            "delta=",
            row["total_delta_voxels_after_minus_v17"],
            "abs_delta=",
            row["total_abs_delta_voxels_after_minus_v17"],
            "zip=",
            row["submission_zip"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
