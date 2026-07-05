#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

UPDATE_ROWDATA="${UPDATE_ROWDATA:-1}"

if [[ "${UPDATE_ROWDATA}" == "1" ]]; then
  run_step "[1/2]" boatrace-backfill-rowdata --rowdata rowdata
else
  echo "[1/2] skipped rowdata update"
fi
run_step "[2/2]" boatrace-build --rowdata rowdata --output data/processed

echo "data_build.sh completed successfully."
