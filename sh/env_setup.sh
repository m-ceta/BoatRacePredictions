#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-boatrace-predictions}"
MINIFORGE_DIR="${MINIFORGE_DIR:-${HOME}/miniforge3}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
INSTALL_RCLONE="${INSTALL_RCLONE:-1}"
CONDA_INIT="${CONDA_INIT:-1}"
CONFIGURE_RCLONE="${CONFIGURE_RCLONE:-1}"
MOUNT_GDRIVE="${MOUNT_GDRIVE:-1}"
ENABLE_SWAP="${ENABLE_SWAP:-1}"
SWAP_FILE="${SWAP_FILE:-/swapfile}"
SWAP_SIZE="${SWAP_SIZE:-16G}"
SWAP_MIN_FREE_GB="${SWAP_MIN_FREE_GB:-25}"
RCLONE_REMOTE_NAME="${RCLONE_REMOTE_NAME:-gdrive}"
RCLONE_REMOTE_PATH="${RCLONE_REMOTE_PATH:-${RCLONE_REMOTE_NAME}:}"
GDRIVE_MOUNT_DIR="${GDRIVE_MOUNT_DIR:-${HOME}/gdrive}"

cd "${PROJECT_ROOT}"

setup_swap() {
  if [[ "${ENABLE_SWAP}" != "1" ]]; then
    echo "[extra] Skipped swap setup"
    return
  fi

  if swapon --show=NAME --noheadings | grep -Fx "${SWAP_FILE}" >/dev/null 2>&1; then
    echo "[extra] Swap already active: ${SWAP_FILE}"
    return
  fi

  if [[ ! -f "${SWAP_FILE}" ]]; then
    swap_dir="$(dirname "${SWAP_FILE}")"
    free_kb="$(df -Pk "${swap_dir}" | awk 'NR==2 {print $4}')"
    required_kb=$((SWAP_MIN_FREE_GB * 1024 * 1024))
    if (( free_kb < required_kb )); then
      echo "[extra] Skipped swap setup: ${swap_dir} has less than ${SWAP_MIN_FREE_GB}GB free"
      return
    fi

    echo "[extra] Creating ${SWAP_SIZE} swap file: ${SWAP_FILE}"
    if ! sudo fallocate -l "${SWAP_SIZE}" "${SWAP_FILE}" 2>/dev/null; then
      swap_size_mb="$(numfmt --from=iec --to-unit=1048576 "${SWAP_SIZE}")"
      sudo dd if=/dev/zero of="${SWAP_FILE}" bs=1M count="${swap_size_mb}" status=progress
    fi
  else
    echo "[extra] Swap file already exists: ${SWAP_FILE}"
  fi

  sudo chmod 600 "${SWAP_FILE}"
  sudo mkswap "${SWAP_FILE}" >/dev/null
  sudo swapon "${SWAP_FILE}"

  if ! grep -F "${SWAP_FILE} none swap sw 0 0" /etc/fstab >/dev/null 2>&1; then
    echo "[extra] Persisting swap in /etc/fstab"
    echo "${SWAP_FILE} none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
}

if [[ "${INSTALL_SYSTEM_PACKAGES}" == "1" ]]; then
  packages=(git curl ca-certificates build-essential zip unzip)
  if [[ "${INSTALL_RCLONE}" == "1" ]]; then
    packages+=(rclone fuse3)
  fi
  echo "[1/5] Installing Debian packages: ${packages[*]}"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
else
  echo "[1/5] Skipped Debian package installation"
fi

setup_swap

if ! command -v conda >/dev/null 2>&1; then
  arch="$(uname -m)"
  case "${arch}" in
    x86_64) miniforge_arch="x86_64" ;;
    aarch64|arm64) miniforge_arch="aarch64" ;;
    *)
      echo "Unsupported architecture for Miniforge: ${arch}" >&2
      exit 1
      ;;
  esac

  installer="/tmp/Miniforge3-Linux-${miniforge_arch}.sh"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${miniforge_arch}.sh"
  echo "[2/5] Installing Miniforge: ${url}"
  curl -L "${url}" -o "${installer}"
  bash "${installer}" -b -p "${MINIFORGE_DIR}"
  rm -f "${installer}"
  # shellcheck disable=SC1091
  source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
else
  echo "[2/5] Conda already exists: $(command -v conda)"
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
fi

if [[ "${CONDA_INIT}" == "1" ]]; then
  echo "[3/5] Initializing conda for bash"
  conda init bash
else
  echo "[3/5] Skipped conda init"
fi

if conda env list | awk '{print $1}' | grep -Fx "${ENV_NAME}" >/dev/null 2>&1; then
  echo "[4/5] Updating conda environment: ${ENV_NAME}"
  conda install -n "${ENV_NAME}" -y -c conda-forge python=3.11 pip
else
  echo "[4/5] Creating conda environment: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" -y -c conda-forge python=3.11 pip
fi

echo "[5/5] Installing project runtime dependencies"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt
conda run -n "${ENV_NAME}" python -m pip install -e .

if [[ "${INSTALL_RCLONE}" == "1" && "${CONFIGURE_RCLONE}" == "1" ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone was not found after package installation." >&2
    exit 1
  fi

  if rclone listremotes | grep -Fx "${RCLONE_REMOTE_NAME}:" >/dev/null 2>&1; then
    echo "[extra] rclone remote already configured: ${RCLONE_REMOTE_NAME}:"
  else
    echo
    echo "[extra] rclone remote \"${RCLONE_REMOTE_NAME}:\" is not configured."
    echo "Create a Google Drive remote. Recommended values:"
    echo "  name: ${RCLONE_REMOTE_NAME}"
    echo "  storage: drive"
    echo "  scope: drive"
    echo "  Use auto config?: n"
    echo
    rclone config
  fi
fi

if [[ "${INSTALL_RCLONE}" == "1" && "${MOUNT_GDRIVE}" == "1" ]]; then
  echo "[extra] Mounting Google Drive"
  RCLONE_REMOTE_PATH="${RCLONE_REMOTE_PATH}" GDRIVE_MOUNT_DIR="${GDRIVE_MOUNT_DIR}" bash "${SCRIPT_DIR}/drive_mount.sh"
fi

echo
echo "Google Cloud minimal conda setup completed."
echo "Restart your shell or run:"
echo "  source ~/.bashrc"
echo "Then activate with:"
echo "  conda activate ${ENV_NAME}"
echo "Google Drive mount directory:"
echo "  ${GDRIVE_MOUNT_DIR}"
