#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${MVAA_INPUT_DIR:-/input}"
OUTPUT_DIR="${MVAA_OUTPUT_DIR:-/output}"
WORK_DIR="${MVAA_WORK_DIR:-/work}"

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"

python /workspace/scripts/final_infer_v39_safe.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --work-dir "${WORK_DIR}"
