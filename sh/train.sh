#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

run_step "[1/2]" boatrace-train --config configs/train.yaml
run_step "[2/2]" boatrace-train-trifecta-v2 --config configs/train.yaml --max-races 1000 --eval-max-races 1000 --eval-rerank-top-n 10

echo "train.sh completed successfully."
