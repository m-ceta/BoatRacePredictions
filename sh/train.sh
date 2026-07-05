#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

MAX_RACES="${MAX_RACES:-1000}"
EVAL_MAX_RACES="${EVAL_MAX_RACES:-3000}"
OPTIMIZE_RERANK="${OPTIMIZE_RERANK:-0}"
RESET_RERANK_OPTIMIZATION="${RESET_RERANK_OPTIMIZATION:-0}"

run_step "[1/2]" boatrace-train --config configs/train.yaml

trifecta_args=(
  --config configs/train.yaml
  --max-races "${MAX_RACES}"
  --eval-max-races "${EVAL_MAX_RACES}"
)
if [[ "${OPTIMIZE_RERANK}" == "1" ]]; then
  trifecta_args+=(--optimize-rerank)
fi
if [[ "${RESET_RERANK_OPTIMIZATION}" == "1" ]]; then
  trifecta_args+=(--reset-rerank-optimization)
fi

run_step "[2/2]" boatrace-train-trifecta-v2 "${trifecta_args[@]}"

echo "train.sh completed successfully."
