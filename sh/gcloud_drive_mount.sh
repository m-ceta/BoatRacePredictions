#!/usr/bin/env bash
set -euo pipefail

RCLONE_REMOTE_PATH="${RCLONE_REMOTE_PATH:-gdrive:}"
GDRIVE_MOUNT_DIR="${GDRIVE_MOUNT_DIR:-${HOME}/gdrive}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone was not found. Install and configure rclone first." >&2
  echo "Example: sudo apt-get update && sudo apt-get install -y rclone fuse3" >&2
  echo "Then run: rclone config" >&2
  exit 1
fi

mkdir -p "${GDRIVE_MOUNT_DIR}"

if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "${GDRIVE_MOUNT_DIR}"; then
  echo "Google Drive is already mounted: ${GDRIVE_MOUNT_DIR}"
  exit 0
fi

echo "Mounting ${RCLONE_REMOTE_PATH} -> ${GDRIVE_MOUNT_DIR}"
rclone mount "${RCLONE_REMOTE_PATH}" "${GDRIVE_MOUNT_DIR}" \
  --daemon \
  --vfs-cache-mode writes \
  --dir-cache-time 1h \
  --poll-interval 1m

echo "Google Drive mounted: ${GDRIVE_MOUNT_DIR}"
echo "Package directory default: ${GDRIVE_MOUNT_DIR}/BoatRacePredictions"
