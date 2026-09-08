# modeltool: DL model → ARM64 shared library

`modeltool.py` is the framework-aware front end.  It keeps conversion,
validation and deployment records together, while reusing
`tools/model_deploy.py` for the already-tested TensorFlow Lite ARM64 linker.

## Commands

```bash
python3 ai_runtime_prototype/modeltool/modeltool.py inspect MODEL
python3 ai_runtime_prototype/modeltool/modeltool.py convert MODEL --output model.tflite
python3 ai_runtime_prototype/modeltool/modeltool.py validate MODEL --tflite model.tflite --output validation.json
python3 ai_runtime_prototype/modeltool/modeltool.py build MODEL --target arm64 --output-dir dist/MODEL_NAME
```

Supported input suffixes are `.tflite`, `.onnx`, `.pt` and `.pth`.  A suffix
only selects the adapter; it is not a claim that every model of that framework
can be converted.

## Artifact layout

`build` creates the following layout:

```text
dist/tiny_gru/
├── source/                    # supplied model and metadata
├── converted/model_float32.tflite
├── reference/                 # optional user-provided input tensor
├── arm64/libtiny_gru.so
├── arm64/elf_report.json
├── manifest.json
└── validation_report.json
```

The application deployed to an ARM64 Linux target needs the resulting `.so`
and the generated `ai_model.h` / `ai_runtime.h` headers at compile time.  The
model and TensorFlow Lite CPU runtime are embedded in the `.so`.

## Create the portable ARM64 Bazel configuration

TensorFlow's default embedded-ARM configuration is generated in Bazel's cache
and therefore records cache paths belonging to one development machine.  Do
not copy that generated directory to another host.  After installing the ARM
GNU toolchain, create a local configuration on each host instead:

```bash
cd ~/qairt_sdk
export ARM64_TOOLCHAIN="$HOME/toolchains/gcc-arm-10.2-aarch64"

python3 ai_runtime_prototype/tools/generate_aarch64_bazel_config.py \
  --tensorflow-root ~/tensorflow \
  --toolchain "$ARM64_TOOLCHAIN" \
  --output "$HOME/toolchains/tf-arm10-bazel-config"
```

The generator detects the installed GCC version, writes paths to the selected
toolchain rather than `~/.cache/bazel`, and creates the `BUILD.bazel` /
`WORKSPACE` files required by Bazel.  It is intentionally ARM64-only; ARM32 is
not configured or tested by this project.

## Generic ARM64 model runner

`tflite_qnn_prototype/examples/ai_model_runner.c` is a reusable client for a
generated `libai_model.so`.  It calls `ai_model_get_io_count()` after
initialisation, so it does not hard-code the legacy QoS model's 240 values or
TinyGRU's 270 values.  It accepts one or more consecutive input tensors from a
raw file or standard input; every tensor is little-endian `float32` and must
follow the model's feature order and preprocessing contract.

Build it beside a generated library:

```bash
"$ARM64_TOOLCHAIN/bin/aarch64-none-linux-gnu-gcc" \
  -std=c11 -O2 -Wall -Wextra \
  tflite_qnn_prototype/examples/ai_model_runner.c \
  -Idist/tiny_gru/arm64 -Ldist/tiny_gru/arm64 \
  -Wl,-rpath,'$ORIGIN' -lai_model \
  -o dist/tiny_gru/arm64/ai_model_runner
```

On the target, place `ai_model_runner` and `libai_model.so` in the same
directory, then run either:

```bash
./ai_model_runner input.raw
producer | ./ai_model_runner -
```

The runner is generic only at the tensor-I/O layer.  Packet parsing, feature
extraction, normalisation and mapping `label` to a traffic class remain the
responsibility of the caller/model contract.

## TinyGRU PyTorch checkpoint

The current verified PyTorch adapter supports the traffic TinyGRU checkpoint
that contains `model_config` and `model_state_dict`.  It rebuilds the inference
graph, loads weights, exports float32 TFLite with LiteRT-Torch, then compares
PyTorch and TFLite outputs on the supplied reference input.

Use a real preprocessed `.npy` tensor, shape `[1,90,3]`, rather than a raw
packet capture:

```bash
cd ~/qairt_sdk
export ARM64_TOOLCHAIN="$HOME/toolchains/gcc-arm-10.2-aarch64"

python3 ai_runtime_prototype/modeltool/modeltool.py build \
  /path/to/tiny_gru_for_tflite.pt \
  --target arm64 \
  --metadata /path/to/preprocessing_metadata.json \
  --reference-input /path/to/input.npy \
  --trust-pytorch-pickle \
  --output-dir dist/tiny_gru \
  --tensorflow-root ~/tensorflow \
  --aarch64-toolchain "$ARM64_TOOLCHAIN" \
  --aarch64-toolchain-config "$HOME/toolchains/tf-arm10-bazel-config"
```

`--trust-pytorch-pickle` is intentionally explicit: loading a non-TorchScript
PyTorch file can execute Python code embedded by its producer.  Do this only
for a model received from a trusted source.

## Generic state_dict `.pt` / `.pth`

If a checkpoint contains weights only, no deployment tool can infer the Python
architecture from tensor names alone.  The model owner supplies a small loader
file, based on [model_loader_template.py](examples/model_loader_template.py),
which exposes either:

```python
def load_model(checkpoint_path: str):
    ...
    return model.eval()
```

or:

```python
def build_model():
    return model
```

Then build with the same command and add:

```bash
--model-loader /path/to/loader.py \
--reference-input /path/to/preprocessed_input.npy
```

The loader is only used on the development host to reconstruct the model.  It
is not included in the target artifact.  Treat it as trusted Python code.

## Current limits

- A state_dict-only PyTorch file requires `--model-loader`; model architecture
  and preprocessing cannot be recovered reliably from weights alone.
- ONNX v1 accepts one input for parity validation.  `onnx2tf` must support all
  operators in the source graph; the tool saves converter logs on failure.
- Existing TFLite is inspected and packaged directly; there is no separate
  source-framework output to compare.
- ARM64 packaging is tested with the iGate-compatible GNU toolchain and target
  GLIBC 2.32.  `elf_report.json` rejects a library requiring a newer GLIBC.
