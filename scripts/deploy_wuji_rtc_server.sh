#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DIR="${OPENWAM_WUJI_CKPT:-${REPO_ROOT}/checkpoints/grasp_anything_base_rtc}"
HOST="${OPENWAM_SERVER_HOST:-0.0.0.0}"
PORT="${OPENWAM_SERVER_PORT:-8848}"
if [[ -n "${OPENWAM_PYTHON:-}" ]]; then
    PYTHON_BIN="${OPENWAM_PYTHON}"
elif [[ -x "/home/ubuntu/miniconda3/envs/openwam/bin/python" ]]; then
    PYTHON_BIN="/home/ubuntu/miniconda3/envs/openwam/bin/python"
else
    PYTHON_BIN="python"
fi

for required in config.yaml normalization_stats.npy; do
    if [[ ! -f "${CKPT_DIR}/${required}" ]]; then
        echo "[ERROR] Missing checkpoint artifact: ${CKPT_DIR}/${required}" >&2
        echo "Wait for the checkpoint download to finish or set OPENWAM_WUJI_CKPT." >&2
        exit 2
    fi
done

if ! nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] NVIDIA driver is not ready; refusing to load the checkpoint." >&2
    if [[ -r /proc/driver/nvidia/version ]]; then
        loaded_version="$(sed -n 's/^NVRM version:.*  \([0-9][0-9.]*\)  Release.*/\1/p' /proc/driver/nvidia/version)"
        installed_version="$(modinfo -F version nvidia 2>/dev/null || true)"
        if [[ -n "${loaded_version}" && -n "${installed_version}" && "${loaded_version}" != "${installed_version}" ]]; then
            echo "[ERROR] Loaded NVIDIA module=${loaded_version}, installed module=${installed_version}." >&2
            echo "[ERROR] Reboot this server, verify 'nvidia-smi', then run this command again." >&2
        else
            echo "[ERROR] Run 'nvidia-smi' for the driver error, fix it, then retry." >&2
        fi
    fi
    exit 3
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/deploy_rtc.py \
    --ckpt-dir "${CKPT_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
