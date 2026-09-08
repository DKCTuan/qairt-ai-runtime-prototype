import sys
from pathlib import Path

from ..core.process import run


def build_shared_library(tflite: Path, output: Path, *, tensorflow_root: Path,
                         toolchain: Path | None, compiler: Path | None,
                         toolchain_config: Path | None,
                         quantize: str, force: bool) -> Path:
    """Delegate ARM64 model-selective library creation to model_deploy.py."""
    deploy = Path(__file__).resolve().parents[2] / "tools" / "model_deploy.py"
    command = [sys.executable, str(deploy), "static-library", "--shared",
               "--model", str(tflite), "--output", str(output),
               "--tensorflow-root", str(tensorflow_root), "--quantize", quantize,
               "--bazel-batch"]
    if toolchain:
        command.extend(["--aarch64-toolchain", str(toolchain)])
    if compiler:
        command.extend(["--aarch64-compiler", str(compiler)])
    if toolchain_config:
        command.extend(["--aarch64-toolchain-config", str(toolchain_config)])
    if force:
        command.append("--force")
    run(command, cwd=deploy.parents[2], log_path=output.with_suffix(".build.log"))
    return output
