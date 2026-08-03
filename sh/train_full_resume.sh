#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${PIPELINE_STATE_DIR:-${PROJECT_ROOT}/.gcloud_pipeline_state}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
activate_conda_env

cd "${PROJECT_ROOT}"
mkdir -p "${STATE_DIR}"

train_args=(--config configs/train.yaml --resume)
if [[ "${SKIP_TRAIN_EVALUATION:-${BOATRACE_TRAIN_SKIP_EVALUATION:-0}}" == "1" ]]; then
  train_args+=(--skip-evaluation)
fi

log_time() {
  TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST'
}

run_pipeline_step_once() {
  local label="$1"
  local marker="$2"
  shift 2
  if [[ -f "${STATE_DIR}/${marker}" ]]; then
    echo "[$(log_time)] ${label} skipped: ${marker} exists"
    return 0
  fi
  echo "[$(log_time)] ${label} $*"
  "$@"
  touch "${STATE_DIR}/${marker}"
}

echo "[$(log_time)] [1/5] drive_mount bash ${SCRIPT_DIR}/drive_mount.sh"
bash "${SCRIPT_DIR}/drive_mount.sh"
touch "${STATE_DIR}/01_drive_mount.done"

run_pipeline_step_once "[2/5] zip_update_local" \
  "02_zip_update_local.done" \
  bash "${SCRIPT_DIR}/zip_update_local.sh"

run_pipeline_step_once "[3/5] build" \
  "03_build.done" \
  boatrace-build --rowdata rowdata --output data/processed

run_pipeline_step_once "[4/5] train" \
  "04_train.done" \
  boatrace-train "${train_args[@]}"

run_pipeline_step_once "[5/5] zip_upload" \
  "05_zip_upload.done" \
  bash "${SCRIPT_DIR}/zip_upload.sh"

echo "train_full_resume.sh completed successfully."
