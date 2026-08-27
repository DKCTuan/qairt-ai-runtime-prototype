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

For models whose converter cannot infer input metadata:

```bash
python3 tools/model_deploy.py prepare model.onnx \
  --source-model-input-shape input "1,224,224,3" \
  --out-tensor-node output
```

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
The current direct-QNN runtime supports exactly one FLOAT32 input and one
FLOAT32 output; unsupported tensor layouts fail explicitly.

## Build backends manually

```bash
make BACKEND=mock run
make BACKEND=qnn_cli run
make BACKEND=qnn_api run
```
