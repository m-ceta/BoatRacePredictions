#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-boatrace-predictions}"

cd "${PROJECT_ROOT}"

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found on PATH." >&2
    return 1
  fi

  local conda_base
  conda_base="$(conda info --base 2>/dev/null)"
  if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    echo "Failed to resolve conda.sh from conda info --base." >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

run_step() {
  local label="$1"
  shift
  echo "${label} $*"
  "$@"
}
