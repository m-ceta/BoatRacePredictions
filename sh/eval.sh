#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

run_step "[1/1]" boatrace-eval-trifecta-full --config configs/train.yaml

echo "eval.sh completed successfully."
