#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

DRIVE_PACKAGE_DIR="${DRIVE_PACKAGE_DIR:-${HOME}/gdrive/gcolab_workdir/btp}"
EXPORT_ROWDATA="${EXPORT_ROWDATA:-1}"
EXPORT_DATA="${EXPORT_DATA:-1}"
EXPORT_ARTIFACTS="${EXPORT_ARTIFACTS:-1}"

mkdir -p "${DRIVE_PACKAGE_DIR}"

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
run_step "[1/1]" boatrace-package-export "${export_args[@]}"

echo "zip_upload.sh completed successfully."
