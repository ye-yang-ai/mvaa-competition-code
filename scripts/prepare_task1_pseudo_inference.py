#!/usr/bin/env python3
"""Prepare high-priority Task1 unlabeled cases for nnU-Net pseudo-label inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("outputs/analysis/task1_pseudo/unlabeled_metadata_screen.csv"),
    )
    parser.add_argument("--priority", type=str, default="high", choices=("high", "medium", "low"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudo/task1_v17_high759"))
    parser.add_argument("--link-mode", type=str, default="symlink", choices=("symlink", "hardlink", "copy"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {args.metadata_csv}")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    images_ts = args.output_dir / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.metadata_csv.open(encoding="utf-8")))
    selected = [row for row in rows if row["metadata_priority"] == args.priority]
    if not selected:
        raise RuntimeError(f"No cases selected for priority={args.priority}")

    manifest = []
    for row in selected:
        case_id = row["case_id"]
        src = Path(row["path"])
        if not src.is_file():
            raise FileNotFoundError(f"Missing unlabeled image: {src}")
        nnunet_case_id = f"pseudo_{case_id}"
        dst = images_ts / f"{nnunet_case_id}_0000.nii.gz"
        link_image(src, dst, args.link_mode)
        manifest.append(
            {
                "case_id": case_id,
                "nnunet_case_id": nnunet_case_id,
                "source_image": src.as_posix(),
                "nnunet_image": dst.as_posix(),
                "metadata_priority": row["metadata_priority"],
                "tukey_fail_count": int(row["tukey_fail_count"]),
                "tukey_failed_fields": row["tukey_failed_fields"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "case_list.txt").write_text(
        "\n".join(item["case_id"] for item in manifest) + "\n", encoding="utf-8"
    )
    (args.output_dir / "nnunet_case_list.txt").write_text(
        "\n".join(item["nnunet_case_id"] for item in manifest) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(json.dumps({"cases": manifest}, indent=2), encoding="utf-8")

    record = [
        "# Task1 pseudo-label inference input",
        "",
        f"- metadata csv: `{args.metadata_csv}`",
        f"- priority: `{args.priority}`",
        f"- selected cases: `{len(manifest)}`",
        f"- link mode: `{args.link_mode}`",
        f"- imagesTs: `{images_ts}`",
        "",
        "nnU-Net case ids use the `pseudo_` prefix to avoid collisions with labeled cases.",
        "",
    ]
    (args.output_dir / "pseudo_inference_record.md").write_text("\n".join(record), encoding="utf-8")

    print(f"Prepared {len(manifest)} cases")
    print(f"imagesTs: {images_ts}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
