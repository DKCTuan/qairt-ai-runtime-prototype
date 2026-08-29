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
The direct-QNN runtime currently supports one input and one output. It reads
their metadata dynamically and supports FLOAT32 plus per-tensor UINT8/INT8
scale-offset quantization. Multi-input/output, dynamic shapes, and per-axis
quantization are rejected explicitly rather than being interpreted wrongly.

## Build backends manually

```bash
make BACKEND=mock run
make BACKEND=qnn_cli run
make BACKEND=qnn_api run
```
