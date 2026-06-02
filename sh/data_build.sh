#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

run_step "[1/3]" boatrace-backfill-rowdata --rowdata rowdata
run_step "[2/3]" boatrace-build --rowdata rowdata --output data/processed
run_step "[3/3]" boatrace-package-upload

echo "data_build.sh completed successfully."
