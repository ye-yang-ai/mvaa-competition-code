#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 FOLD GPU_ID" >&2
  exit 2
fi

FOLD="$1"
GPU_ID="$2"
ROOT="/home/wuyongji/yangye_project/MVAA_v1"
RUN_ROOT="$ROOT/outputs/nnunet_task1_pseudo_top50"
LOG_DIR="$RUN_ROOT/train_logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="/home/wuyongji/miniconda3/envs/mvaa/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="/tmp/mvaa_mpl_nnunet"
export nnUNet_raw="$RUN_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$RUN_ROOT/nnUNet_preprocessed"
export nnUNet_results="$RUN_ROOT/nnUNet_results"

cd "$ROOT"
exec > >(tee -a "$LOG_DIR/fold_${FOLD}_gpu_${GPU_ID}.log") 2>&1

date '+%F %T %Z'
echo "Starting Dataset113_MVAA_Task1_PseudoTop50 fold ${FOLD} on GPU ${GPU_ID}"
/home/wuyongji/miniconda3/envs/mvaa/bin/nnUNetv2_train 113 3d_fullres "$FOLD"
