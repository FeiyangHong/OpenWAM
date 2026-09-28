#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTRIBOT_SDK="${ASTRIBOT_SDK_ROOT:-/home/astribot/Documents/astribot_sdk_aarch64}"
WUJI_ROOT="${WUJI_SDK_ROOT:-/home/astribot/wuji-sdk}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

source_required() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Required environment file not found: $1" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    set +u
    source "$1"
    set -u
}

source_required "${ROS_SETUP}"
source_required "${ASTRIBOT_SDK}/env.sh"
if [[ -f "${WUJI_ROOT}/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "${WUJI_ROOT}/install/setup.bash"
    set -u
fi

export PYTHONPATH="${SCRIPT_DIR}:${WUJI_ROOT}/.venv/lib/python3.10/site-packages:/home/astribot/.local/lib/python3.10/site-packages:/usr/local/lib/python3.10/dist-packages:${ASTRIBOT_SDK}:${ASTRIBOT_SDK}/third_party/software/astribot_ros_middleware/lib/python3.10/site-packages:${ASTRIBOT_SDK}/astribot_msgs/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ASTRIBOT_SDK}/astribot_msgs/lib:${ASTRIBOT_SDK}/astribot_sdk/core/common/robotics_library_py:${ASTRIBOT_SDK}/astribot_sdk/core/common/whole_body_control/third_party:${ASTRIBOT_SDK}/third_party/drake/lib:${ASTRIBOT_SDK}/third_party/third_pkg:${LD_LIBRARY_PATH:-}"

cd "${SCRIPT_DIR}"
exec python3 -m openwam_robot_client.rtc_main "$@"
