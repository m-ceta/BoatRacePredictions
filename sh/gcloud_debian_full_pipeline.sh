#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${PIPELINE_STATE_DIR:-${PROJECT_ROOT}/.gcloud_pipeline_state}"
MAX_RACES="${MAX_RACES:-1000}"
EVAL_MAX_RACES="${EVAL_MAX_RACES:-3000}"

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

run_pipeline_step "[1/5] gcloud_drive_mount" \
  "01_gcloud_drive_mount.done" \
  bash "${SCRIPT_DIR}/gcloud_drive_mount.sh"

run_pipeline_step "[2/5] gcloud_drive_restore_update" \
  "02_gcloud_drive_restore_update.done" \
  bash "${SCRIPT_DIR}/gcloud_drive_restore_update.sh"

run_pipeline_step "[3/5] data_build" \
  "03_data_build.done" \
  env UPDATE_ROWDATA=0 bash "${SCRIPT_DIR}/data_build.sh"

run_pipeline_step "[4/5] train with fresh rerank optimization" \
  "04_train.done" \
  env \
    MAX_RACES="${MAX_RACES}" \
    EVAL_MAX_RACES="${EVAL_MAX_RACES}" \
    OPTIMIZE_RERANK=1 \
    RESET_RERANK_OPTIMIZATION=1 \
    bash "${SCRIPT_DIR}/train.sh"

run_pipeline_step "[5/5] gcloud_build_train_upload export only" \
  "05_gcloud_build_train_upload.done" \
  env \
    RUN_BUILD=0 \
    RUN_BASE_TRAIN=0 \
    RUN_TRIFECTA_TRAIN=0 \
    RUN_EXPORT=1 \
    bash "${SCRIPT_DIR}/gcloud_build_train_upload.sh"

echo "gcloud_debian_full_pipeline.sh completed successfully."
