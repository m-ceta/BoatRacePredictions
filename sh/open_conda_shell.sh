#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
activate_conda_env

echo "Activated conda environment \"${ENV_NAME}\"."
echo "Project root: ${PROJECT_ROOT}"
echo "boatrace-* commands are available in this shell."

exec "${SHELL:-/bin/bash}" -i
