# Standalone TFLite CPU payload

`standalone-tflite` produces one statically linked ARM64 executable.  The
`.tflite` model, AI Runtime API, and TensorFlow Lite C runtime are compiled
into that ELF.  The generated application includes only `ai_runtime.h`; it
does not include or call TensorFlow Lite. A target therefore needs no model
file, shared library, dynamic loader bundle, QNN SDK, package installation, or
network/API service at inference time.

This is the CPU deployment route for a generic ARM64 Linux target, such as the
iGate test device.  It is intentionally separate from the QAIRT/QNN route:
QNN/HTP requires target-specific Qualcomm BSP drivers and cannot be made
portable by embedding it in this executable.

## Build on the host

TensorFlow 2.15 source and Bazelisk must already be available.  For the ARM64
toolchain configured in this project:

```bash
python3 ai_runtime_prototype/tools/model_deploy.py standalone-tflite \
  --model ~/model_test/tflite_output/traffic_qos_model_float32.tflite \
  --output dist/traffic_runner \
  --tensorflow-root ~/tensorflow
```

The tool rejects a non-`.tflite` model, embeds the model bytes in C source,
builds `//tensorflow/lite/c:c_api` with `--config=elinux_aarch64`, and verifies
with `readelf` that the output has no dynamic dependency.  `--force` replaces
an existing output; `--keep-build-dir` retains the generated temporary Bazel
package for diagnosis.

## Optional application source using only AI Runtime API

The default executable is a generic raw-tensor runner. To compile an actual
embedded application whose `main()` knows no TensorFlow API, pass its C source:

```bash
python3 ai_runtime_prototype/tools/model_deploy.py standalone-tflite \
  --model ~/model_test/tflite_output/traffic_qos_model_float32.tflite \
  --app-source tflite_qnn_prototype/examples/standalone_classifier_app.c \
  --output dist/traffic_classifier
```

The example has the intended boundary: `dl_init()`, `dl_get_io_count()`,
`dl_inference()` and `dl_deinit()`. It is intentionally a float32,
single-input classifier example. Quantized, multi-input/output, detector, or
regression models should use the default generic runner and `dl_runtime_*` raw
tensor API instead.

## Run on target

Only the executable and the live/preprocessed input are required.  For a
single-input model:

```sh
chmod +x /tmp/traffic_runner
mkdir -p /tmp/results
/tmp/traffic_runner --output-dir /tmp/results /tmp/input_seq.raw
```

For a model with multiple inputs, provide one raw file per input in model input
order.  Each raw file must match the tensor's exact dtype, shape and byte size.
The program writes `output_0.raw`, `output_1.raw`, ... and prints latency JSON.
Preprocessing (for example JPEG decode, resize, normalization) and semantic
postprocessing (for example class names, NMS) remain application-specific.
