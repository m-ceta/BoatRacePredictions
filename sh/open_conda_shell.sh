#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"
activate_conda_env

if ! command -v boatrace-train >/dev/null 2>&1; then
  echo "boatrace-train was not found after activating \"${ENV_NAME}\"." >&2
  echo "Reinstall project commands with:" >&2
  echo "  conda run -n ${ENV_NAME} python -m pip install -e ${PROJECT_ROOT}" >&2
fi

echo "Activated conda environment \"${ENV_NAME}\"."
echo "Project root: ${PROJECT_ROOT}"
echo "boatrace-* commands are available in this shell."

conda_base="$(conda info --base 2>/dev/null)"
rcfile="$(mktemp)"
{
  echo '[[ -f ~/.bashrc ]] && source ~/.bashrc'
  printf 'source %q\n' "${conda_base}/etc/profile.d/conda.sh"
  printf 'conda activate %q\n' "${ENV_NAME}"
  printf 'cd %q\n' "${PROJECT_ROOT}"
  printf 'rm -f %q\n' "${rcfile}"
} >"${rcfile}"

exec bash --rcfile "${rcfile}" -i
