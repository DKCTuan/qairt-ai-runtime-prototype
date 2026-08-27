#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

export DL_QAIRT_ROOT="${DL_QAIRT_ROOT:-/home/congtuan/qairt_sdk/qairt/2.44.0.260225}"
export DL_MODEL_PATH="${DL_MODEL_PATH:-/home/congtuan/model_test/traffic_qos_model.dlc}"
export DL_INPUT_RAW="${DL_INPUT_RAW:-/home/congtuan/model_test/input_seq.raw}"
export DL_WORK_DIR="${DL_WORK_DIR:-/tmp/ai_runtime_qnn}"
export DL_INPUT_NAME="${DL_INPUT_NAME:-input_seq}"
export DL_OUTPUT_NAME="${DL_OUTPUT_NAME:-class_probs}"

make BACKEND=qnn_cli run
