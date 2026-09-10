# Model conversion on the QSDK build server

`modeltool` has two separate jobs: conversion on the x86_64 build host and
TFLite-to-ARM64-musl linking with TensorFlow/Bazel. Neither PyTorch, ONNX nor
Python is installed on the deployment target.

## Required virtual environments

Use Python 3.10 or newer. On the current build server `python3` may resolve to
an older interpreter, so always pass the venv executable explicitly.

```text
venvs/model-torch/bin/python  numpy, torch, litert-torch
venvs/model-onnx/bin/python   numpy, tensorflow, onnx, onnxruntime, onnx2tf
venvs/tf215/bin/python        numpy, tensorflow
```

Run the preflight before a conversion:

```bash
python "$REPO_ROOT/ai_runtime_prototype/tools/check_model_conversion_environment.py" \
  --torch-python "$BUILD_ROOT/venvs/model-torch/bin/python" \
  --onnx-python "$BUILD_ROOT/venvs/model-onnx/bin/python" \
  --tflite-python "$BUILD_ROOT/venvs/tf215/bin/python" \
  --tensorflow-root "$TF_ROOT" \
  --aarch64-compiler "$ARM64_CC"
```

Record `pip freeze` from each working environment in the model release record
before upgrading any package. This repository intentionally does not claim
that arbitrary latest PyTorch, LiteRT Torch, TensorFlow and ONNX packages are
compatible.

## PyTorch and ONNX models

For a deployable conversion provide the source model, model contract and a
real preprocessed reference tensor for every input. A state-dict-only `.pt`/`.pth` additionally
needs a trusted loader that reconstructs the architecture.

```text
model.pt | model.pth | model.onnx
preprocessing metadata / class order
one reference `.npy` per input
loader.py (state_dict-only PyTorch only)
```

`modeltool build` stores the converted TFLite model, source input and
validation record alongside the ARM64 library.

## Multi-input models such as TinyGRU P16

Multi-input conversion uses `model_profile.json` as the ordered ABI contract.
Pass every real, already-preprocessed reference tensor by name:

```bash
python "$REPO_ROOT/ai_runtime_prototype/modeltool/modeltool.py" convert \
  /path/to/checkpoint.pt \
  --output /path/to/model.tflite \
  --model-profile "$REPO_ROOT/traffic_models_p16/model_profiles/tinygru_p16/model_profile.json" \
  --reference-input numeric=/path/to/reference_numeric.npy \
  --reference-input p16_ids=/path/to/reference_p16_ids.npy \
  --model-loader /path/to/p16_loader.py \
  --trust-pytorch-pickle \
  --torch-python "$BUILD_ROOT/venvs/model-torch/bin/python" \
  --tflite-python "$BUILD_ROOT/venvs/tf215/bin/python"
```

The tool compares every source and converted output and rejects a TFLite whose
input/output shape or dtype differs from the profile. A state-dict checkpoint
still needs trusted architecture code in `--model-loader`; tensor metadata
cannot reconstruct an arbitrary model class.
