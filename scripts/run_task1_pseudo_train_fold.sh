#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="${1:?dataset id required}"
FOLD="${2:?fold required}"
GPU_ID="${3:?gpu id required}"
NNUNET_ROOT="${4:?nnU-Net root required}"

REPO_ROOT="/home/wuyongji/yangye_project/MVAA_v1"
LOG_DIR="${REPO_ROOT}/${NNUNET_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LD_LIBRARY_PATH="/home/wuyongji/miniconda3/envs/mvaa/lib"
export MPLCONFIGDIR="/tmp/mvaa_mpl_nnunet"
export nnUNet_raw="${REPO_ROOT}/${NNUNET_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/${NNUNET_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/${NNUNET_ROOT}/nnUNet_results"

cd "${REPO_ROOT}"

{
  date
  echo "dataset_id=${DATASET_ID} fold=${FOLD} gpu=${GPU_ID} nnunet_root=${NNUNET_ROOT}"
  /home/wuyongji/miniconda3/envs/mvaa/bin/nnUNetv2_train \
    "${DATASET_ID}" \
    3d_fullres \
    "${FOLD}"
  date
} 2>&1 | tee "${LOG_DIR}/train_dataset${DATASET_ID}_fold${FOLD}_gpu${GPU_ID}.log"
