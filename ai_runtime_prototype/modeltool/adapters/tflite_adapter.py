import json
import subprocess
from pathlib import Path

from ..core.errors import ModelToolError


_PROBE = r'''
import json, sys
import numpy as np
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=sys.argv[1])
interpreter.allocate_tensors()
def desc(item):
    return {"name": item["name"], "shape": item["shape"].tolist(),
            "dtype": np.dtype(item["dtype"]).name}
print(json.dumps({"inputs": [desc(x) for x in interpreter.get_input_details()],
                  "outputs": [desc(x) for x in interpreter.get_output_details()]}))
'''


def inspect_tflite(source: Path, *, python: Path, log_path: Path) -> dict:
    if not python.is_file():
        raise ModelToolError("TFLITE_ENVIRONMENT_MISSING", f"TFLite Python not found: {python}")
    result = subprocess.run([str(python), "-c", _PROBE, str(source)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ModelToolError("TFLITE_INVALID", f"cannot inspect TFLite model; see: {log_path}")
    return json.loads(result.stdout)
