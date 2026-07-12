#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${PIPELINE_STATE_DIR:-${PROJECT_ROOT}/.gcloud_pipeline_state}"
MAX_RACES="${MAX_RACES:-10000}"
EVAL_MAX_RACES="${EVAL_MAX_RACES:-3000}"
OPTIMIZE_RERANK_WORKERS="${OPTIMIZE_RERANK_WORKERS:-2}"

cd "${PROJECT_ROOT}"
rm -rf "${STATE_DIR}"
mkdir -p "${STATE_DIR}"

log_time() {
  TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST'
}

run_pipeline_step() {
  local label="$1"
  local marker="$2"
  shift 2
  echo "[$(log_time)] ${label} $*"
  "$@"
  touch "${STATE_DIR}/${marker}"
}

run_train_with_fresh_rerank_optimization() {
  boatrace-train --config configs/train.yaml
  local train_args=(
    --config configs/train.yaml
    --max-races "${MAX_RACES}"
    --eval-max-races "${EVAL_MAX_RACES}"
    --optimize-rerank
    --optimize-rerank-workers "${OPTIMIZE_RERANK_WORKERS}"
    --reset-rerank-optimization
  )
  boatrace-train-trifecta-v2 "${train_args[@]}"
}

run_pipeline_step "[1/6] drive_mount" \
  "01_drive_mount.done" \
  bash "${SCRIPT_DIR}/drive_mount.sh"

run_pipeline_step "[2/6] zip_update_local" \
  "02_zip_update_local.done" \
  bash "${SCRIPT_DIR}/zip_update_local.sh"

run_pipeline_step "[3/6] build" \
  "03_build.done" \
  boatrace-build --rowdata rowdata --output data/processed

run_pipeline_step "[4/6] train with fresh rerank optimization" \
  "04_train.done" \
  run_train_with_fresh_rerank_optimization

run_pipeline_step "[5/6] full trifecta evaluation" \
  "05_full_trifecta_evaluation.done" \
  boatrace-eval-trifecta-full --config configs/train.yaml

run_pipeline_step "[6/6] zip_upload" \
  "06_zip_upload.done" \
  bash "${SCRIPT_DIR}/zip_upload.sh"

echo "train_full.sh completed successfully."
