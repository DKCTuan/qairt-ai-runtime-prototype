#!/usr/bin/env bash
set -euo pipefail

# Sinh QNN context binary (.bin) tu mot .dlc da convert/validate, dung mot
# lan tren host/CI. File .bin sinh ra la thu duoc dong goi cung firmware
# khi deploy len thiet bi nhung - luc do runtime chi con contextCreateFromBinary().
#
# Usage:
#   ./scripts/build_context_binary.sh [output.bin]
#
# Bien moi truong:
#   DL_QAIRT_ROOT   thu muc goc QAIRT SDK
#   DL_MODEL_PATH   duong dan .dlc dau vao

DL_QAIRT_ROOT="${DL_QAIRT_ROOT:-/home/congtuan/qairt_sdk/qairt/2.44.0.260225}"
DL_MODEL_PATH="${DL_MODEL_PATH:-/home/congtuan/model_test/traffic_qos_model.dlc}"
OUTPUT_BIN="${1:-${DL_MODEL_PATH%.dlc}.bin}"

GEN_BIN="$DL_QAIRT_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator"
BACKEND_LIB="$DL_QAIRT_ROOT/lib/x86_64-linux-clang/libQnnCpu.so"
MODEL_LIB="$DL_QAIRT_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so"
LIB_DIR="$DL_QAIRT_ROOT/lib/x86_64-linux-clang"

echo "model      : $DL_MODEL_PATH"
echo "output     : $OUTPUT_BIN"
echo "qairt_root : $DL_QAIRT_ROOT"

LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
"$GEN_BIN" \
  --backend "$BACKEND_LIB" \
  --model "$MODEL_LIB" \
  --dlc_path "$DL_MODEL_PATH" \
  --binary_file "$(basename "$OUTPUT_BIN" .bin)" \
  --output_dir "$(dirname "$OUTPUT_BIN")"

echo "done: $OUTPUT_BIN"
