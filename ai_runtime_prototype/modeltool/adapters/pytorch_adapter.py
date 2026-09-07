import json
import sys
from pathlib import Path

from ..core.errors import ModelToolError
from ..core.process import run


def convert_pytorch(source: Path, output: Path, *, torch_python: Path,
                    metadata: Path | None, input_shape: str | None,
                    reference_input: Path | None, model_loader: Path | None,
                    trust_pickle: bool) -> dict:
    """Use the maintained TinyGRU/TorchScript exporter in model-torch venv."""
    if not torch_python.is_file():
        raise ModelToolError("PYTORCH_ENVIRONMENT_MISSING",
                             f"model-torch Python not found: {torch_python}")
    exporter = Path(__file__).resolve().parents[2] / "tools" / "pytorch_to_tflite.py"
    command = [str(torch_python), str(exporter), "--model", str(source),
               "--output", str(output)]
    if metadata:
        command.extend(["--metadata", str(metadata)])
    if input_shape:
        command.extend(["--input-shape", input_shape])
    if reference_input:
        command.extend(["--reference-input", str(reference_input)])
    if model_loader:
        command.extend(["--model-loader", str(model_loader)])
    if trust_pickle:
        command.append("--trust-pytorch-pickle")
    run(command, log_path=output.with_suffix(".pytorch.log"),
        extra_env={"CUDA_VISIBLE_DEVICES": ""})
    report_path = output.with_suffix(output.suffix + ".export.json")
    if not report_path.is_file():
        raise ModelToolError("CONVERSION_FAILED", f"missing exporter report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))
