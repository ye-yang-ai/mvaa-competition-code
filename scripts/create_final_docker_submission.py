#!/usr/bin/env python3
"""Create the final CodaBench Docker descriptor submission.zip."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Docker/OCI image reference, preferably pinned by digest.")
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--output-zip", type=Path, default=Path("outputs/final_docker_submission/submission.zip"))
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a copy of submission.json.")
    return parser.parse_args()


def validate_image_ref(image: str) -> None:
    if not image or image.strip() != image:
        raise ValueError("Image reference is empty or has leading/trailing whitespace.")
    if any(ord(ch) < 32 for ch in image):
        raise ValueError("Image reference contains control characters.")
    if any(ch.isspace() for ch in image):
        raise ValueError("Image reference must not contain whitespace.")


def main() -> int:
    args = parse_args()
    validate_image_ref(args.image)
    if args.timeout_seconds <= 0 or args.timeout_seconds > 43200:
        raise ValueError("--timeout-seconds must be in the range 1..43200.")

    payload = {
        "image": args.image,
        "timeout_seconds": int(args.timeout_seconds),
    }
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json_bytes)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_bytes(json_bytes)

    print(f"Wrote {args.output_zip}")
    print("Zip root: submission.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
