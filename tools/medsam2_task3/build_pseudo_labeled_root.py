#!/usr/bin/env python3
"""Build a Task3 labeled-root directory from original labels plus filtered pseudo masks."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-root", type=Path, default=REPO_ROOT / "data/reference_data/t3_vid/train")
    parser.add_argument(
        "--filter-report",
        type=Path,
        default=REPO_ROOT / "outputs/pseudo/task3_medsam2/filtered_masks_debug/reports/filter_report.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/pseudo/task3_medsam2/pseudo_labeled_train_debug",
    )
    parser.add_argument("--source-image-root", type=Path, default=None, help="Root for pseudo source images; defaults to labeled-root.")
    parser.add_argument("--copy-original", action="store_true", default=True)
    parser.add_argument("--no-copy-original", action="store_false", dest="copy_original")
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    return parser.parse_args()


def copy_tree_files(src_root: Path, dst_root: Path) -> int:
    count = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count


def read_kept_rows(filter_report: Path) -> list[dict[str, str]]:
    with filter_report.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if str(row.get("keep", "0")) == "1"]
    return rows


def main() -> int:
    args = parse_args()
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    original_files = 0
    if args.copy_original:
        original_files = copy_tree_files(args.labeled_root, args.output_root)

    pseudo_files = 0
    pseudo_samples = 0
    missing_source = 0
    seen_pseudo_keys: set[tuple[str, int]] = set()
    source_image_root = args.source_image_root or args.labeled_root
    rows = read_kept_rows(args.filter_report)
    for row in rows:
        src_image = Path(row["raw_mask_path"])
        # raw_mask_path is not the image. Use original image path by deriving from report is not possible,
        # so locate it from the source video/frame naming convention in the original labeled root.
        source_video_id = row["source_video_id"]
        source_frame_idx = int(row["source_frame_idx"])
        image_name = f"{source_video_id}_{source_frame_idx:06d}.png"
        src_image = source_image_root / source_video_id / image_name
        src_mask = Path(row["filtered_mask_path"])
        if not src_image.exists() or not src_mask.exists():
            missing_source += 1
            continue
        key = (source_video_id, source_frame_idx)
        if key in seen_pseudo_keys:
            continue
        seen_pseudo_keys.add(key)

        dst_dir = args.output_root / "_medsam2_pseudo" / source_video_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_image = dst_dir / image_name
        dst_mask = dst_dir / f"{source_video_id}_{source_frame_idx:06d}_label_bin.png"
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_mask, dst_mask)
        pseudo_files += 2
        pseudo_samples += 1

    summary_path = args.output_root / "_medsam2_pseudo_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"labeled_root={args.labeled_root}",
                f"filter_report={args.filter_report}",
                f"source_image_root={source_image_root}",
                f"output_root={args.output_root}",
                f"original_files_copied={original_files}",
                f"pseudo_file_copies={pseudo_files}",
                f"pseudo_samples={pseudo_samples}",
                f"missing_source_rows={missing_source}",
                f"duplicate_kept_rows_skipped={len(rows) - pseudo_samples - missing_source}",
                f"kept_rows={len(rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
