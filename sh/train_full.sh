#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${PIPELINE_STATE_DIR:-${PROJECT_ROOT}/.gcloud_pipeline_state}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
activate_conda_env

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

run_pipeline_step "[1/5] drive_mount" \
  "01_drive_mount.done" \
  bash "${SCRIPT_DIR}/drive_mount.sh"

run_pipeline_step "[2/5] zip_update_local" \
  "02_zip_update_local.done" \
  bash "${SCRIPT_DIR}/zip_update_local.sh"

run_pipeline_step "[3/5] build" \
  "03_build.done" \
  boatrace-build --rowdata rowdata --output data/processed

run_pipeline_step "[4/5] train" \
  "04_train.done" \
  boatrace-train --config configs/train.yaml

run_pipeline_step "[5/5] zip_upload" \
  "05_zip_upload.done" \
  bash "${SCRIPT_DIR}/zip_upload.sh"

echo "train_full.sh completed successfully."
