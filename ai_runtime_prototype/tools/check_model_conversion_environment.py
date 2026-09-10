#!/usr/bin/env python3
"""Report whether a build host is ready for modeltool conversion and ARM64 build.

This deliberately imports each framework in its own configured interpreter so
the result reflects the actual venv that modeltool will use, not the shell's
default Python.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def probe_python(label, executable, modules):
    executable = Path(executable).expanduser()
    result = {"label": label, "python": str(executable), "ok": False}
    if not executable.is_file():
        result["error"] = "Python executable not found"
        return result
    script = """
import importlib.metadata as metadata
import json, sys
modules = %s
details = {}
for module, distribution in modules:
    __import__(module)
    details[module] = metadata.version(distribution)
print(json.dumps({"python_version": sys.version.split()[0], "packages": details}))
""" % repr(modules)
    process = subprocess.run([str(executable), "-c", script], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=False)
    if process.returncode:
        result["error"] = process.stderr.strip() or process.stdout.strip()
        return result
    result.update(json.loads(process.stdout))
    result["ok"] = True
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check isolated PyTorch, ONNX and TFLite conversion environments")
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--onnx-python", required=True)
    parser.add_argument("--tflite-python", required=True)
    parser.add_argument("--tensorflow-root", required=True)
    parser.add_argument("--aarch64-compiler", required=True)
    args = parser.parse_args()

    checks = [
        probe_python("model-torch", args.torch_python,
                     [("numpy", "numpy"), ("torch", "torch"),
                      ("litert_torch", "litert-torch")]),
        probe_python("model-onnx", args.onnx_python,
                     [("numpy", "numpy"), ("tensorflow", "tensorflow"),
                      ("onnx", "onnx"), ("onnxruntime", "onnxruntime"),
                      ("onnx2tf", "onnx2tf")]),
        probe_python("tflite-inspection", args.tflite_python,
                     [("numpy", "numpy"), ("tensorflow", "tensorflow")]),
    ]
    tf_root = Path(args.tensorflow_root).expanduser()
    compiler = Path(args.aarch64_compiler).expanduser()
    checks.append({"label": "tensorflow-source", "path": str(tf_root),
                   "ok": (tf_root / "WORKSPACE").is_file()})
    checks.append({"label": "aarch64-compiler", "path": str(compiler),
                   "ok": compiler.is_file()})
    report = {"status": "PASS" if all(item["ok"] for item in checks) else "FAIL",
              "checks": checks}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
