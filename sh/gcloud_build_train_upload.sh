#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

DRIVE_PACKAGE_DIR="${DRIVE_PACKAGE_DIR:-${HOME}/gdrive/BoatRacePredictions}"
MAX_RACES="${MAX_RACES:-1000}"
EVAL_MAX_RACES="${EVAL_MAX_RACES:-3000}"
OPTIMIZE_RERANK="${OPTIMIZE_RERANK:-1}"
RESET_RERANK_OPTIMIZATION="${RESET_RERANK_OPTIMIZATION:-0}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_BASE_TRAIN="${RUN_BASE_TRAIN:-1}"
RUN_TRIFECTA_TRAIN="${RUN_TRIFECTA_TRAIN:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
EXPORT_ROWDATA="${EXPORT_ROWDATA:-1}"
EXPORT_DATA="${EXPORT_DATA:-1}"
EXPORT_ARTIFACTS="${EXPORT_ARTIFACTS:-1}"

mkdir -p "${DRIVE_PACKAGE_DIR}"

if [[ "${RUN_BUILD}" == "1" ]]; then
  run_step "[1/4]" boatrace-build --rowdata rowdata --output data/processed
else
  echo "[1/4] skipped boatrace-build"
fi

if [[ "${RUN_BASE_TRAIN}" == "1" ]]; then
  run_step "[2/4]" boatrace-train --config configs/train.yaml
else
  echo "[2/4] skipped boatrace-train"
fi

train_args=(
  --config configs/train.yaml
  --max-races "${MAX_RACES}"
  --eval-max-races "${EVAL_MAX_RACES}"
)
if [[ "${OPTIMIZE_RERANK}" == "1" ]]; then
  train_args+=(--optimize-rerank)
fi
if [[ "${RESET_RERANK_OPTIMIZATION}" == "1" ]]; then
  train_args+=(--reset-rerank-optimization)
fi
if [[ "${RUN_TRIFECTA_TRAIN}" == "1" ]]; then
  run_step "[3/4]" boatrace-train-trifecta-v2 "${train_args[@]}"
else
  echo "[3/4] skipped boatrace-train-trifecta-v2"
fi

export_args=(--project-root . --output-dir "${DRIVE_PACKAGE_DIR}")
if [[ "${EXPORT_ROWDATA}" == "0" ]]; then
  export_args+=(--skip-rowdata)
fi
if [[ "${EXPORT_DATA}" == "0" ]]; then
  export_args+=(--skip-data)
fi
if [[ "${EXPORT_ARTIFACTS}" == "0" ]]; then
  export_args+=(--skip-artifacts)
fi
if [[ "${RUN_EXPORT}" == "1" ]]; then
  run_step "[4/4]" boatrace-package-export "${export_args[@]}"
else
  echo "[4/4] skipped boatrace-package-export"
fi

echo "gcloud_build_train_upload.sh completed successfully."
