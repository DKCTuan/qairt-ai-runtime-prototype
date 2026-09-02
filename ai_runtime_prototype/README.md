# AI Runtime Prototype for Qualcomm QAIRT/QNN

The public C API keeps applications independent from Qualcomm tooling:

```c
dl_init();
dl_inference_ex(input, &result);
dl_deinit();
```

The main runtime path uses a prebuilt QNN context binary and calls the QNN C
API directly. `qnn_cli` remains available for comparison and diagnostics.

## Simplest model workflow

Convert ONNX or TFLite to DLC. The output defaults next to the input model:

```bash
python3 tools/model_deploy.py convert /path/to/model.onnx
```

Convert and build a CPU context binary in one command:

```bash
python3 tools/model_deploy.py prepare /path/to/model.onnx
```

Choose an output directory or backend only when needed:

```bash
python3 tools/model_deploy.py prepare /path/to/model.onnx \
  --work-dir /path/to/artifacts \
  --backend htp
```

The tool finds QAIRT from `DL_QAIRT_ROOT`, `QAIRT_ROOT`, or `QNN_SDK_ROOT`.
If none is set, it discovers the newest SDK below the adjacent `qairt/`
directory. Override explicitly with global `--qairt-root` when necessary.

Run a read-only preflight before conversion, especially when preparing a
device target or HTP artifact:

```bash
python3 tools/model_deploy.py doctor /path/to/model.tflite --backend htp
```

For models whose converter cannot infer input metadata:

```bash
python3 tools/model_deploy.py prepare model.onnx \
  --source-model-input-shape input "1,224,224,3" \
  --out-tensor-node output
```

## PyTorch input

QAIRT includes `qnn-pytorch-converter`, but it loads a **TorchScript** archive
(`torch.jit.load`), not a training checkpoint/state dictionary. Export the
trained model with its Python class available, for example:

```python
model.eval()
example = torch.zeros(1, 240, dtype=torch.float32)
torch.jit.save(torch.jit.trace(model, example), "traffic_model.ts.pt")
```

Then convert it directly (the input name and dimensions must match the model):

```bash
python3 tools/model_deploy.py prepare traffic_model.ts.pt \
  --pytorch-input-dim input 1,240
```

The QAIRT Python environment must contain a PyTorch version compatible with
the SDK. If conversion fails or the model has unsupported operators, export to
ONNX and use the ONNX path; keep a reference output to validate both paths.

Uncommon Qualcomm converter options can be forwarded after `--extra-args`.

## Convert, run, and validate

Run the full host CPU pipeline and optionally compare with a reference:

```bash
python3 tools/model_deploy.py deploy \
  --model /path/to/model.onnx \
  --input-raw /path/to/input.raw \
  --reference /path/to/reference.npy
```

`run-api` executes locally and currently supports the x86_64 QAIRT target.
`prepare` should be used to generate artifacts intended for another device.
The instance-based `dl_runtime_*` API and direct-QNN/TFLite backends support
multiple input and output tensors. Tensor metadata is discovered dynamically,
with FLOAT32 plus per-tensor UINT8/INT8 scale-offset quantization. The legacy
`dl_inference*` convenience API remains one-input/one-output. Dynamic shapes
are rejected. QNN per-axis quantized tensors can execute through the raw API;
their metadata reports the quantized axis and `scale=0` because the compact
public struct does not expose the full scale array.

Inspect a Context Binary without preparing input data or running inference:

```bash
python3 tools/model_deploy.py inspect-context \
  --context /path/to/model.bin --backend cpu
```

Run it through the generic raw-tensor API. Repeat `--input-raw` in graph
tensor order for multi-input models; each output is written separately:

```bash
python3 tools/model_deploy.py run-context \
  --context /path/to/model.bin \
  --input-raw input_0.raw --input-raw input_1.raw \
  --output-dir /path/to/outputs --backend cpu
```

The raw-tensor API recognizes FLOAT16/FLOAT32/FLOAT64, signed and unsigned
8/16/32/64-bit integers, and BOOL8 where exposed by the backend. The legacy
float convenience API intentionally accepts only FLOAT32 and per-tensor
UINT8/INT8.

Validation accepts quantized raw outputs and uses combined absolute/relative
tolerance. For example:

```bash
python3 tools/model_deploy.py validate \
  --dlc model.dlc --input-raw input.raw --reference reference.npy \
  --output-dtype int8 --output-scale 0.03125 --output-zero-point 0 \
  --tolerance 1e-3 --relative-tolerance 1e-2
```

## Package for an Embedded Linux board

Qualcomm HTP deployment needs application-processor libraries (`libQnnHtp.so`,
`libQnnSystem.so`, and an architecture-matched Stub) as well as a matching
Hexagon Skel. Create a self-contained bundle before copying anything to the
board. This command does **not** connect to or modify a device:

```bash
python3 tools/model_deploy.py package-target \
  --artifact artifacts/model.htp.bin \
  --artifact-kind context-binary \
  --backend htp --htp-arch v73 \
  --target aarch64-oe-linux-gcc11.2 \
  --app /path/to/aarch64/traffic_app \
  --output /path/to/model-v73-bundle
```

Read `manifest.json` before deployment. Copy the complete directory to the
board, then run `./run.sh` there. The target ABI, HTP architecture, BSP
firmware, driver and Context Binary must all match; packaging alone cannot
prove HTP graph delegation. The manifest includes SHA-256 checksums for every
payload file. `package-target` currently packages the direct-QNN Context
Binary path; DLC and TFLite use different entrypoints and runtime dependencies.

The optional REST server mirrors `doctor`, `inspect-context`, `run-context`,
conversion and validation. Requests are serialized because converter/runtime
jobs may otherwise collide in shared work directories.

## Build backends manually

```bash
make BACKEND=mock run
make BACKEND=qnn_cli run
make BACKEND=qnn_api run
```
