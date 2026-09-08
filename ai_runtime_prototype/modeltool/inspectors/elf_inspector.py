import re
import subprocess
from pathlib import Path

from ..core.errors import ModelToolError


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.removeprefix("GLIBC_").split("."))


def inspect_shared_library(path: Path, *, target_libc: str = "glibc",
                           target_glibc: str | None = None) -> dict:
    """Check architecture and whether its libc ABI matches the selected target."""
    commands = {
        "header": ["readelf", "-h", str(path)],
        "dynamic": ["readelf", "-d", str(path)],
        "versions": ["readelf", "--version-info", str(path)],
    }
    text = {}
    for name, command in commands.items():
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise ModelToolError("ABI_CHECK_FAILED", f"{' '.join(command)} failed: {result.stdout}")
        text[name] = result.stdout
    machine = re.search(r"Machine:\s*(.+)", text["header"])
    if not machine or "AArch64" not in machine.group(1):
        found = machine.group(1).strip() if machine else "unknown"
        raise ModelToolError("ABI_CHECK_FAILED", f"expected AArch64 shared library, got: {found}")
    dependencies = re.findall(r"Shared library: \[([^]]+)\]", text["dynamic"])
    versions = sorted(set(re.findall(r"GLIBC_\d+(?:\.\d+)+", text["versions"])),
                      key=_version_tuple)
    highest = versions[-1] if versions else None
    report = {
        "architecture": "AArch64",
        "dependencies": dependencies,
        "glibc_max": highest,
        "target_libc": target_libc,
        "status": "PASS",
    }
    if target_libc == "musl":
        if highest:
            raise ModelToolError(
                "ABI_CHECK_FAILED",
                f"musl target cannot load library requiring {highest}")
        return report
    if target_libc != "glibc":
        raise ModelToolError("ABI_CHECK_FAILED", f"unsupported target libc: {target_libc}")
    if target_glibc and highest:
        required = _version_tuple(highest)
        target = _version_tuple(target_glibc if target_glibc.startswith("GLIBC_")
                                else f"GLIBC_{target_glibc}")
        report["target_glibc"] = ".".join(map(str, target))
        if required > target:
            raise ModelToolError("ABI_CHECK_FAILED",
                                 f"library requires {highest}, target has GLIBC_{'.'.join(map(str, target))}")
    return report
