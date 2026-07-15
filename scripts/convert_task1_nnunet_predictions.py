#!/usr/bin/env python3
"""Convert nnU-Net Task1 predictions to MVAA submission format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-json", type=Path, default=Path("outputs/nnunet_task1_fold0/task1_case_mapping.json"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--submission-task-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    out_dir = args.submission_task_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} is not empty. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for item in mapping["test"]:
        case_id = item["case_id"]
        src = args.pred_dir / f"{case_id}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"Missing nnU-Net prediction: {src}")
        dst = out_dir / f"{case_id}-pred.nii.gz"
        shutil.copy2(src, dst)
        records.append({"case_id": case_id, "segmentation": dst.name})

    output_json = out_dir / "task1_predictions.json"
    output_json.write_text(json.dumps({"cases": records}, indent=2), encoding="utf-8")
    print(f"Converted {len(records)} predictions to {out_dir}")
    print(f"Saved json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
