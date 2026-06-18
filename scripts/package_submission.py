#!/usr/bin/env python3
"""Package MVAA validation predictions with POSIX paths for CodaBench."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


TASK_DIRS = ("t1_ct", "t2_tee", "t3_vid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MVAA submission.zip.")
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path("outputs/quick/submission"),
        help="Directory containing t1_ct, t2_tee, and t3_vid.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("outputs/quick/submission.zip"),
        help="Output zip path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.submission_dir
    output_zip = args.output_zip

    if not root.exists():
        raise FileNotFoundError(f"Submission directory not found: {root}")

    missing = [task for task in TASK_DIRS if not (root / task).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing task directories under {root}: {', '.join(missing)}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    written = 0
    with ZipFile(output_zip, "w", ZIP_DEFLATED) as zf:
        for task in TASK_DIRS:
            task_dir = root / task
            for path in sorted(task_dir.rglob("*")):
                if not path.is_file():
                    continue
                arcname = path.relative_to(root).as_posix()
                zf.write(path, arcname)
                written += 1

    print(f"Saved: {output_zip}")
    print(f"Files: {written}")
    print("First entries:")
    with ZipFile(output_zip) as zf:
        for name in zf.namelist()[:30]:
            print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
