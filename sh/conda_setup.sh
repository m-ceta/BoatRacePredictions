#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-boatrace-predictions}"

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found on PATH." >&2
  exit 1
fi

if conda env list | grep -E "^[*[:space:]]*${ENV_NAME}[[:space:]]" >/dev/null 2>&1; then
  echo "Updating conda environment \"${ENV_NAME}\" from environment.yml..."
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "Creating conda environment \"${ENV_NAME}\" from environment.yml..."
  conda env create -f environment.yml
fi

echo
echo "Setup completed."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
