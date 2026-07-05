#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_conda_env

DRIVE_PACKAGE_DIR="${DRIVE_PACKAGE_DIR:-${HOME}/gdrive/BoatRacePredictions}"
RESTORE_ROWDATA="${RESTORE_ROWDATA:-1}"
RESTORE_DATA="${RESTORE_DATA:-1}"
RESTORE_ARTIFACTS="${RESTORE_ARTIFACTS:-1}"
UPDATE_ROWDATA="${UPDATE_ROWDATA:-1}"

if [[ ! -d "${DRIVE_PACKAGE_DIR}" ]]; then
  echo "Drive package directory was not found: ${DRIVE_PACKAGE_DIR}" >&2
  exit 1
fi

restore_args=(--project-root . --source-dir "${DRIVE_PACKAGE_DIR}")
if [[ "${RESTORE_ROWDATA}" == "0" ]]; then
  restore_args+=(--skip-rowdata)
fi
if [[ "${RESTORE_DATA}" == "0" ]]; then
  restore_args+=(--skip-data)
fi
if [[ "${RESTORE_ARTIFACTS}" == "0" ]]; then
  restore_args+=(--skip-artifacts)
fi

run_step "[1/2]" boatrace-package-restore-local "${restore_args[@]}"

if [[ "${UPDATE_ROWDATA}" == "1" ]]; then
  run_step "[2/2]" boatrace-backfill-rowdata --rowdata rowdata
else
  echo "[2/2] skipped rowdata update"
fi

echo "gcloud_drive_restore_update.sh completed successfully."
