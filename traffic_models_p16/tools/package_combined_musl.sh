#!/usr/bin/env bash
# Package the unified Random Forest + TinyGRU P16 musl SDK for handoff.
set -euo pipefail

usage() {
    echo "usage: $0 --sdk-dir DIR --target-root DIR --output-root DIR [--model-profile FILE]" >&2
    exit 2
}

sdk_dir= target_root= output_root= model_profile=
while (($#)); do
    case "$1" in
        --sdk-dir) sdk_dir=${2:?}; shift 2 ;;
        --target-root) target_root=${2:?}; shift 2 ;;
        --output-root) output_root=${2:?}; shift 2 ;;
        --model-profile) model_profile=${2:?}; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n $sdk_dir && -n $target_root && -n $output_root ]] || usage

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
for file in "$sdk_dir/libtraffic_ai.so" "$sdk_dir/app_arm64"; do
    [[ -f $file ]] || { echo "error: missing artifact: $file" >&2; exit 1; }
done
model_libraries=()
for file in "$sdk_dir"/lib*.so; do
    [[ -f $file && $(basename "$file") != libtraffic_ai.so ]] && model_libraries+=("$file")
done
[[ ${#model_libraries[@]} -eq 1 ]] || {
    echo "error: sdk-dir must contain exactly one model .so beside libtraffic_ai.so" >&2
    exit 1
}
if [[ -z $model_profile ]]; then
    model_profile="$repo_root/traffic_models_p16/model_profiles/tinygru_p16/model_profile.json"
fi
[[ -f $model_profile ]] || { echo "error: missing model profile: $model_profile" >&2; exit 1; }

mkdir -p "$output_root"
name="traffic_ai_qsdk_musl_unified_rf_p16_$(date +%Y%m%d_%H%M%S)"
package="$output_root/$name"
mkdir -p "$package/bin" "$package/include" "$package/lib" "$package/docs"

cp "$sdk_dir/libtraffic_ai.so" "${model_libraries[0]}" "$package/lib/"
cp "$sdk_dir/app_arm64" "$package/bin/"
cp "$repo_root/traffic_models_p16/traffic_p16.h" "$package/include/"
cp "$(dirname "$model_profile")/traffic_p16_deployment_config.h" "$package/include/"
cp "$repo_root/traffic_models_p16/traffic_ai.h" "$package/include/"
cp "$repo_root/traffic_models_p16/model_contract.h" "$package/include/"
cp "$model_profile" "$package/docs/model_profile.json"
cp "$repo_root/traffic_models_p16/HANDOFF.md" "$package/docs/"
cp "$repo_root/traffic_models_p16/MODEL_PROVENANCE.md" "$package/docs/"

if [[ -f "$target_root/lib/libgcc_s.so.1" ]]; then
    cp -L "$target_root/lib/libgcc_s.so.1" "$package/lib/"
fi

(cd "$package" && sha256sum bin/* include/* lib/* docs/* > SHA256SUMS)
tar -C "$output_root" -czf "$package.tar.gz" "$name"
echo "folder=$package"
echo "archive=$package.tar.gz"
