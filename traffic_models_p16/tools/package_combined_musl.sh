#!/usr/bin/env bash
# Package the unified Random Forest + TinyGRU P16 musl SDK for handoff.
set -euo pipefail

usage() {
    echo "usage: $0 --sdk-dir DIR --target-root DIR --output-root DIR" >&2
    exit 2
}

sdk_dir= target_root= output_root=
while (($#)); do
    case "$1" in
        --sdk-dir) sdk_dir=${2:?}; shift 2 ;;
        --target-root) target_root=${2:?}; shift 2 ;;
        --output-root) output_root=${2:?}; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n $sdk_dir && -n $target_root && -n $output_root ]] || usage

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
for file in \
    "$sdk_dir/libtraffic_ai.so" \
    "$sdk_dir/libtiny_gru_p16_float32.so" \
    "$sdk_dir/app_traffic_ai_arm64"; do
    [[ -f $file ]] || { echo "error: missing artifact: $file" >&2; exit 1; }
done

mkdir -p "$output_root"
name="traffic_ai_qsdk_musl_unified_rf_p16_$(date +%Y%m%d_%H%M%S)"
package="$output_root/$name"
mkdir -p "$package/bin" "$package/include" "$package/lib" "$package/docs"

cp "$sdk_dir/libtraffic_ai.so" "$sdk_dir/libtiny_gru_p16_float32.so" "$package/lib/"
cp "$sdk_dir/app_traffic_ai_arm64" "$package/bin/"
cp "$repo_root/traffic_models/traffic_models.h" "$package/include/"
cp "$repo_root/traffic_models_p16/traffic_p16.h" "$package/include/"
cp "$repo_root/traffic_models_p16/generated/traffic_p16_deployment_config.h" "$package/include/"
cp "$repo_root/traffic_models_p16/traffic_ai.h" "$package/include/"
cp "$repo_root/traffic_models_p16/HANDOFF.md" "$package/docs/"
cp "$repo_root/traffic_models_p16/MODEL_PROVENANCE.md" "$package/docs/"
cp "$repo_root/traffic_models/README.md" "$repo_root/traffic_models/DUAL_MODELS.md" "$package/docs/"

if [[ -f "$target_root/lib/libgcc_s.so.1" ]]; then
    cp -L "$target_root/lib/libgcc_s.so.1" "$package/lib/"
fi

(cd "$package" && sha256sum bin/* include/* lib/* docs/* > SHA256SUMS)
tar -C "$output_root" -czf "$package.tar.gz" "$name"
echo "folder=$package"
echo "archive=$package.tar.gz"
