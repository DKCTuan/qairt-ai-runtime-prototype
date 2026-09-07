"""Compare deployed ARM64 GRU scores with host TFLite, not training accuracy.

Requires numpy and tensorflow in the test environment (not on the board).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--app", required=True)
parser.add_argument("--sysroot", required=True)
args = parser.parse_args()
interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=1)
interpreter.allocate_tensors()
inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
assert len(inputs) == len(outputs) == 1
assert list(inputs[0]["shape"]) == [1, 90, 3]
assert list(outputs[0]["shape"]) == [1, 5]
assert inputs[0]["dtype"] == outputs[0]["dtype"] == np.float32
rng = np.random.default_rng(42)
samples = [np.zeros((1, 90, 3), np.float32),
           rng.normal(size=(1, 90, 3)).astype(np.float32),
           np.full((1, 90, 3), 0.5, np.float32)]
with tempfile.TemporaryDirectory(prefix="gru-parity-") as temp:
    sample_path = Path(temp) / "input.raw"
    for index, sample in enumerate(samples):
        sample.astype("<f4").tofile(sample_path)
        interpreter.set_tensor(inputs[0]["index"], sample)
        interpreter.invoke()
        reference = interpreter.get_tensor(outputs[0]["index"]).reshape(-1)
        proc = subprocess.run(["qemu-aarch64", "-L", args.sysroot,
                               str(Path(args.app).resolve()), "gru", str(sample_path)],
                              text=True, capture_output=True, check=True)
        fields = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        actual = np.array([float(x) for x in fields["scores"].split(",")])
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
        assert int(fields["label"]) == int(reference.argmax())
        print(json.dumps({"sample": index, "label": int(fields["label"]),
                          "max_abs_error": float(np.max(np.abs(actual-reference))),
                          "status": "pass"}))
    sample_path.write_bytes(b"bad")
    proc = subprocess.run(["qemu-aarch64", "-L", args.sysroot,
                           str(Path(args.app).resolve()), "gru", str(sample_path)],
                          text=True, capture_output=True)
    assert proc.returncode != 0 and "1080 bytes" in proc.stderr
