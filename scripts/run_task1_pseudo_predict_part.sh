#!/usr/bin/env bash
set -euo pipefail

PART_ID="${1:?part id required}"
NUM_PARTS="${2:?num parts required}"
GPU_ID="${3:?gpu id required}"

REPO_ROOT="/home/wuyongji/yangye_project/MVAA_v1"
INPUT_DIR="${REPO_ROOT}/outputs/pseudo/task1_v17_high759/imagesTs"
OUTPUT_DIR="${REPO_ROOT}/outputs/pseudo/task1_v17_high759/pred"
LOG_DIR="${REPO_ROOT}/outputs/pseudo/task1_v17_high759/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LD_LIBRARY_PATH="/home/wuyongji/miniconda3/envs/mvaa/lib"
export MPLCONFIGDIR="/tmp/mvaa_mpl_nnunet"
export nnUNet_raw="${REPO_ROOT}/outputs/nnunet_task1_fold0/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/outputs/nnunet_task1_fold0/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/outputs/nnunet_task1_fold0/nnUNet_results"

cd "${REPO_ROOT}"

{
  date
  echo "part_id=${PART_ID} num_parts=${NUM_PARTS} gpu=${GPU_ID}"
  /home/wuyongji/miniconda3/envs/mvaa/bin/nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -d 101 \
    -c 3d_fullres \
    -f 0 1 2 3 4 \
    -chk checkpoint_best.pth \
    -device cuda \
    --save_probabilities \
    --continue_prediction \
    --disable_progress_bar \
    -num_parts "${NUM_PARTS}" \
    -part_id "${PART_ID}" \
    -npp 1 \
    -nps 1
  date
} 2>&1 | tee "${LOG_DIR}/part_${PART_ID}_of_${NUM_PARTS}_gpu_${GPU_ID}.log"
