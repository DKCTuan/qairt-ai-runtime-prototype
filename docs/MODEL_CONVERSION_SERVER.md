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

## One-input PyTorch and ONNX models

For a deployable conversion provide the source model, model contract and a
real preprocessed reference tensor. A state-dict-only `.pt`/`.pth` additionally
needs a trusted loader that reconstructs the architecture.

```text
model.pt | model.pth | model.onnx
preprocessing metadata / class order
reference_input.npy
loader.py (state_dict-only PyTorch only)
```

`modeltool build` stores the converted TFLite model, source input and
validation record alongside the ARM64 library.

## Multi-input models such as TinyGRU P16

The current generic PyTorch and ONNX adapters validate one source input only.
They must not be used to convert P16 directly. P16 conversion remains:

```text
trusted P16 training/notebook export -> tiny_gru_p16_float32.tflite
                                      -> modeltool build -> ARM64 musl .so
```

The generic TFLite-to-musl build and generated model ABI do support P16's two
TFLite inputs. Extending source-framework conversion needs a tested two-input
reference contract (numeric `.npy` plus P16-ID `.npy`) and is a separate task.
