import json
import shutil
from pathlib import Path

from ..core.errors import ModelToolError
from ..core.process import run


def inspect_onnx(source: Path, *, python: Path, output: Path) -> dict:
    if not python.is_file():
        raise ModelToolError("ONNX_ENVIRONMENT_MISSING", f"ONNX Python not found: {python}")
    script = r'''
import json, sys
import onnx
model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
def shape(value):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in value.type.tensor_type.shape.dim]
def desc(value):
    tensor = value.type.tensor_type
    return {"name": value.name, "onnx_elem_type": tensor.elem_type, "shape": shape(value)}
print(json.dumps({"opset": [x.version for x in model.opset_import],
                  "inputs": [desc(x) for x in model.graph.input],
                  "outputs": [desc(x) for x in model.graph.output]}))
'''
    import subprocess
    result = subprocess.run([str(python), "-c", script, str(source)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout + result.stderr, encoding="utf-8", errors="replace")
    if result.returncode:
        if "ir_version" in result.stderr and "higher than the checker" in result.stderr:
            raise ModelToolError(
                "ONNX_ENVIRONMENT_TOO_OLD",
                "the ONNX file uses a newer IR version than this ONNX environment; "
                f"upgrade the isolated ONNX environment, then retry (log: {output})",
            )
        raise ModelToolError("ONNX_INVALID", f"ONNX validation failed; see: {output}")
    return json.loads(result.stdout)


def convert_onnx(source: Path, output: Path, *, converter: Path,
                 input_shapes: list[str], log_path: Path) -> dict:
    if not converter.is_file():
        raise ModelToolError("ONNX_ENVIRONMENT_MISSING", f"onnx2tf not found: {converter}")
    work = output.with_suffix(".onnx2tf-work")
    if work.exists():
        shutil.rmtree(work)
    command = [str(converter), "-i", str(source), "-o", str(work), "-nuo", "--non_verbose"]
    if input_shapes:
        command.extend(["-ois", *input_shapes])
    run(command, log_path=log_path, extra_env={"CUDA_VISIBLE_DEVICES": ""})
    candidates = sorted(work.rglob("*.tflite"))
    if not candidates:
        raise ModelToolError("CONVERSION_FAILED", f"onnx2tf produced no .tflite; see: {log_path}")
    preferred = [path for path in candidates if "float32" in path.name.lower()]
    selected = preferred[0] if len(preferred) == 1 else candidates[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, output)
    shutil.rmtree(work, ignore_errors=True)
    return {"converter": "onnx2tf", "log": str(log_path)}


def validate_onnx_tflite(source: Path, tflite: Path, reference_inputs: list[str], *,
                          python: Path, output: Path,
                          reference_names: list[str] | None = None) -> dict:
    """Run multi-input ONNX Runtime and TFLite inference on identical tensors."""
    if not reference_inputs:
        raise ModelToolError("REFERENCE_INPUT_REQUIRED", "at least one reference input is required")
    script = r'''
import json, sys
import numpy as np
import onnxruntime as ort
import tensorflow as tf
onnx_path, tflite_path, encoded_inputs, encoded_names = sys.argv[1:]
arguments = json.loads(encoded_inputs)
profile_names = json.loads(encoded_names)
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
source_inputs = session.get_inputs()
named, positional = {}, []
for argument in arguments:
    if "=" in argument:
        name, path = argument.split("=", 1)
        if not name or not path or name in named:
            raise RuntimeError("invalid or duplicate named reference input: " + argument)
        named[name] = path
    else:
        positional.append(argument)
if named and positional:
    raise RuntimeError("do not mix named and positional reference inputs")
if named:
    ordered_names = profile_names or [item.name for item in source_inputs]
    expected = set(ordered_names)
    if set(named) != expected:
        raise RuntimeError("reference names differ from input contract: expected=" +
                           str(sorted(expected)) + " got=" + str(sorted(named)))
    paths = [named[name] for name in ordered_names]
else:
    if len(positional) != len(source_inputs):
        raise RuntimeError("reference input count differs from ONNX input count")
    paths = positional
values = [np.load(path, allow_pickle=False) for path in paths]
source_output = session.run(None, {item.name: value for item, value in zip(source_inputs, values)})
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
if len(inputs) != len(values) or len(outputs) != len(source_output):
    raise RuntimeError("TFLite I/O count differs from ONNX output contract")
for detail, value in zip(inputs, values):
    expected_shape = tuple(int(x) for x in detail["shape"])
    if tuple(value.shape) != expected_shape:
        raise RuntimeError("reference shape differs from TFLite input " + detail["name"] +
                           ": expected=" + str(expected_shape) + " got=" + str(value.shape))
    interpreter.set_tensor(detail["index"], value.astype(detail["dtype"], copy=False))
interpreter.invoke()
tflite_output = [interpreter.get_tensor(x["index"]) for x in outputs]
errors = [np.abs(a - b) for a, b in zip(source_output, tflite_output)]
max_abs = max(float(x.max()) for x in errors)
max_rel = max(float((x / np.maximum(np.abs(a), 1e-12)).max())
              for a, x in zip(source_output, errors))
passed = all(np.allclose(a, b, rtol=1e-4, atol=1e-5)
             for a, b in zip(source_output, tflite_output))
print(json.dumps({"status": "PASS" if passed else "FAIL", "inputs": paths,
                  "max_abs_error": max_abs, "max_rel_error": max_rel,
                  "rtol": 1e-4, "atol": 1e-5}))
'''
    import subprocess
    result = subprocess.run([str(python), "-c", script, str(source), str(tflite),
                             json.dumps(reference_inputs), json.dumps(reference_names or [])],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout + result.stderr, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ModelToolError("PARITY_FAILED", f"ONNX/TFLite validation failed; see: {output}")
    report = json.loads(result.stdout)
    if report["status"] != "PASS":
        raise ModelToolError("PARITY_FAILED", f"ONNX/TFLite output mismatch; see: {output}")
    return report
