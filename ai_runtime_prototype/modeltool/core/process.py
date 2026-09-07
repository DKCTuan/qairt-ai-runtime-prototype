import os
import subprocess
from pathlib import Path

from .errors import ModelToolError


def run(command: list[str], *, log_path: Path, cwd: Path | None = None,
        extra_env: dict[str, str] | None = None) -> None:
    """Run a converter, retain its complete log, and return a stable error."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ModelToolError(
            "CONVERSION_FAILED",
            f"command failed (exit {result.returncode}); see log: {log_path}",
        )
