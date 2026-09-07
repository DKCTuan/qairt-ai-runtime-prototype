#!/usr/bin/env python3
"""Universal DL deployment front end for supported TFLite/ONNX/PyTorch models.

The tool owns inspection, conversion records, validation records and artifact
layout.  It delegates only the ARM64 `.tflite -> .so` link step to the proven
`model_deploy.py static-library --shared` backend.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeltool.adapters.onnx_adapter import (convert_onnx, inspect_onnx,
                                             validate_onnx_tflite)
from modeltool.adapters.pytorch_adapter import convert_pytorch
from modeltool.adapters.tflite_adapter import inspect_tflite
from modeltool.builders.arm64_tflite_builder import build_shared_library
from modeltool.core.detector import detect_model_type
from modeltool.core.errors import ModelToolError
from modeltool.core.manifest import read_json, write_json
from modeltool.inspectors.elf_inspector import inspect_shared_library


def path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def executable(value: str | None) -> Path | None:
    """Keep a venv launcher symlink intact instead of resolving to /usr/bin."""
    return Path(value).expanduser() if value else None


def source_manifest(source: Path, kind: str) -> dict:
    return {
        "name": source.stem,
        "source_model": str(source),
        "source_framework": {"pytorch": "pytorch", "onnx": "onnx", "tflite": "tflite"}[kind],
        "source_format": source.suffix.lower().removeprefix("."),
    }


def apply_tflite_contract(manifest: dict, io: dict) -> None:
    """Promote the common one-input/one-output contract to stable fields."""
    manifest["tflite_io"] = io
    if len(io["inputs"]) == 1:
        item = io["inputs"][0]
        manifest["input_shape"] = item["shape"]
        manifest["input_dtype"] = item["dtype"]
    if len(io["outputs"]) == 1:
        item = io["outputs"][0]
        manifest["output_shape"] = item["shape"]
        manifest["output_dtype"] = item["dtype"]
    metadata = manifest.get("preprocessing")
    if isinstance(metadata, dict) and isinstance(metadata.get("class_names"), list):
        manifest["classes"] = metadata["class_names"]


def inspect_source(args, source: Path, kind: str, log_dir: Path) -> dict:
    manifest = source_manifest(source, kind)
    if kind == "onnx":
        onnx = inspect_onnx(source, python=executable(args.onnx_python),
                            output=log_dir / "onnx_inspect.log")
        manifest["source_io"] = onnx
    elif kind == "tflite":
        apply_tflite_contract(manifest, inspect_tflite(
            source, python=executable(args.tflite_python), log_path=log_dir / "tflite_inspect.log"))
    else:
        # Loading arbitrary pickle data merely to inspect it is unsafe.  The
        # trusted load happens only during conversion with an explicit flag.
        metadata = path(args.metadata)
        if metadata:
            try:
                manifest["preprocessing"] = read_json(metadata)
            except (OSError, json.JSONDecodeError) as error:
                raise ModelToolError("METADATA_INVALID", f"cannot read metadata {metadata}: {error}")
        manifest["load_policy"] = (
            "TorchScript is attempted first. Non-TorchScript input requires "
            "--trust-pytorch-pickle; state_dict-only checkpoints require --model-loader."
        )
    return manifest


def convert_source(args, source: Path, kind: str, tflite: Path, work: Path) -> tuple[dict, dict]:
    """Convert one source to TFLite and return conversion + validation data."""
    if kind == "tflite":
        shutil.copy2(source, tflite)
        return {"status": "PASS", "conversion": "none"}, {
            "status": "NOT_APPLICABLE",
            "reason": "input was already TFLite; no source-framework parity exists",
        }
    if kind == "pytorch":
        report = convert_pytorch(
            source, tflite,
            torch_python=executable(args.torch_python), metadata=path(args.metadata),
            input_shape=args.input_shape, reference_input=path(args.reference_input),
            model_loader=path(args.model_loader),
            trust_pickle=args.trust_pytorch_pickle,
        )
        if report.get("validation_input") == "synthetic_zeros" and not args.allow_synthetic_validation:
            raise ModelToolError(
                "REFERENCE_INPUT_REQUIRED",
                "PyTorch conversion used a synthetic zero tensor. Supply --reference-input INPUT.npy "
                "for a deployable parity result, or explicitly pass --allow-synthetic-validation.",
            )
        validation = {
            "status": "PASS",
            "source": "pytorch",
            "input": report.get("validation_input"),
            "max_abs_error": report["max_abs_error"],
            "rtol": 1e-4,
            "atol": 1e-5,
        }
        return report, validation
    conversion = convert_onnx(
        source, tflite, converter=executable(args.onnx2tf),
        input_shapes=args.onnx_input_shape or [], log_path=work / "onnx2tf.log",
    )
    reference = path(args.reference_input)
    if not reference:
        raise ModelToolError("REFERENCE_INPUT_REQUIRED",
                             "ONNX conversion requires --reference-input INPUT.npy for parity validation")
    validation = validate_onnx_tflite(
        source, tflite, reference, python=executable(args.onnx_python),
        output=work / "onnx_tflite_validation.log",
    )
    return conversion, validation


def cmd_inspect(args) -> None:
    source = path(args.model)
    if not source.is_file():
        raise ModelToolError("MODEL_NOT_FOUND", f"model not found: {source}")
    kind = detect_model_type(source)
    destination = path(args.output) if args.output else source.with_suffix(source.suffix + ".manifest.json")
    manifest = inspect_source(args, source, kind, destination.parent)
    write_json(destination, manifest)
    print(json.dumps({"status": "success", "manifest": str(destination), **manifest}, indent=2))


def cmd_convert(args) -> None:
    source = path(args.model)
    if not source.is_file():
        raise ModelToolError("MODEL_NOT_FOUND", f"model not found: {source}")
    kind = detect_model_type(source)
    output = path(args.output)
    if output.suffix.lower() != ".tflite":
        raise ModelToolError("INVALID_OUTPUT", "convert --output must end in .tflite")
    if output.exists() and not args.force:
        raise ModelToolError("OUTPUT_EXISTS", f"output exists: {output}; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    conversion, validation = convert_source(args, source, kind, output, output.parent)
    manifest = inspect_source(args, source, kind, output.parent)
    manifest["tflite_model"] = str(output)
    apply_tflite_contract(manifest, inspect_tflite(
        output, python=executable(args.tflite_python),
        log_path=output.with_suffix(".inspect.log")))
    manifest["conversion"] = conversion
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    write_json(output.with_suffix(output.suffix + ".validation_report.json"), validation)
    print(json.dumps({"status": "success", "tflite_model": str(output),
                      "validation": validation}, indent=2))


def cmd_validate(args) -> None:
    source, tflite = path(args.model), path(args.tflite)
    if not source.is_file() or not tflite.is_file():
        raise ModelToolError("MODEL_NOT_FOUND", "--model and --tflite must both exist")
    kind = detect_model_type(source)
    report_path = path(args.output)
    if kind == "onnx":
        reference = path(args.reference_input)
        if not reference:
            raise ModelToolError("REFERENCE_INPUT_REQUIRED", "ONNX validation requires --reference-input INPUT.npy")
        report = validate_onnx_tflite(source, tflite, reference,
                                      python=executable(args.onnx_python), output=report_path)
    elif kind == "pytorch":
        raise ModelToolError(
            "PYTORCH_RECONVERT_REQUIRED",
            "PyTorch validation is performed during `convert`/`build`; provide --reference-input there "
            "so the trusted model is loaded once and the generated TFLite is checked.",
        )
    else:
        report = {"status": "NOT_APPLICABLE", "reason": "TFLite has no separate source framework"}
        write_json(report_path, report)
    print(json.dumps(report, indent=2))


def cmd_build(args) -> None:
    source = path(args.model)
    if not source.is_file():
        raise ModelToolError("MODEL_NOT_FOUND", f"model not found: {source}")
    if args.target != "arm64":
        raise ModelToolError("UNSUPPORTED_TARGET", "v1 currently supports only --target arm64")
    kind = detect_model_type(source)
    root = path(args.output_dir) or Path("dist") / source.stem
    if root.exists() and any(root.iterdir()) and not args.force:
        raise ModelToolError("OUTPUT_EXISTS", f"artifact directory exists: {root}; use --force to replace files")
    source_dir, converted_dir, reference_dir, arm64_dir = (
        root / "source", root / "converted", root / "reference", root / "arm64")
    for directory in (source_dir, converted_dir, reference_dir, arm64_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, source_dir / source.name)
    if args.metadata:
        metadata = path(args.metadata)
        if not metadata.is_file():
            raise ModelToolError("METADATA_INVALID", f"metadata not found: {metadata}")
        shutil.copy2(metadata, source_dir / metadata.name)
    reference = path(args.reference_input)
    if reference:
        if not reference.is_file():
            raise ModelToolError("REFERENCE_INPUT_REQUIRED", f"reference input not found: {reference}")
        shutil.copy2(reference, reference_dir / reference.name)

    tflite = converted_dir / "model_float32.tflite"
    conversion, validation = convert_source(args, source, kind, tflite, converted_dir)
    manifest = inspect_source(args, source, kind, root)
    manifest.update({"tflite_model": str(tflite), "conversion": conversion, "target": "arm64"})
    apply_tflite_contract(manifest, inspect_tflite(
        tflite, python=executable(args.tflite_python),
        log_path=converted_dir / "tflite_inspect.log"))
    write_json(root / "validation_report.json", validation)
    write_json(root / "manifest.json", manifest)

    library = arm64_dir / f"lib{source.stem}.so"
    build_shared_library(tflite, library, tensorflow_root=path(args.tensorflow_root),
                         toolchain=path(args.aarch64_toolchain),
                         toolchain_config=path(args.aarch64_toolchain_config),
                         quantize=args.quantize, force=True)
    elf = inspect_shared_library(library, target_glibc=args.target_glibc)
    write_json(arm64_dir / "elf_report.json", elf)
    print(json.dumps({"status": "success", "artifact_dir": str(root),
                      "library": str(library), "manifest": str(root / "manifest.json"),
                      "validation_report": str(root / "validation_report.json"),
                      "elf": elf}, indent=2))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata", help="Preprocessing metadata JSON, if available")
    parser.add_argument("--reference-input", help="Reference input tensor in .npy format")
    parser.add_argument("--input-shape", help="Fallback model input shape, e.g. 1,90,3")
    parser.add_argument("--torch-python", default=str(Path.home() / "venvs/model-torch/bin/python"))
    parser.add_argument("--onnx-python", default=str(Path.home() / "venvs/model-onnx/bin/python"))
    parser.add_argument("--tflite-python", default=str(Path.home() / "venvs/model-onnx/bin/python"))
    parser.add_argument("--onnx2tf", default=str(Path.home() / "venvs/model-onnx/bin/onnx2tf"))
    parser.add_argument("--onnx-input-shape", action="append",
                        help="onnx2tf static input override; repeat for multiple inputs")
    parser.add_argument("--trust-pytorch-pickle", action="store_true",
                        help="Allow loading a trusted non-TorchScript .pt/.pth file")
    parser.add_argument("--model-loader",
                        help="Trusted loader.py exposing load_model(path) or build_model()")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Build validated ARM64 TFLite model libraries")
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Detect a model format and write a manifest")
    p.add_argument("model")
    p.add_argument("--output", help="Manifest path; default: beside source model")
    add_common(p); p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("convert", help="Convert supported source formats to TFLite")
    p.add_argument("model"); p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-synthetic-validation", action="store_true")
    add_common(p); p.set_defaults(func=cmd_convert)
    p = sub.add_parser("validate", help="Validate an existing source/TFLite pair")
    p.add_argument("model"); p.add_argument("--tflite", required=True); p.add_argument("--output", required=True)
    add_common(p); p.set_defaults(func=cmd_validate)
    p = sub.add_parser("build", help="Convert, validate, package and inspect one ARM64 .so")
    p.add_argument("model"); p.add_argument("--target", default="arm64")
    p.add_argument("--output-dir")
    p.add_argument("--tensorflow-root", default=str(Path.home() / "tensorflow"))
    p.add_argument("--aarch64-toolchain")
    p.add_argument("--aarch64-toolchain-config")
    p.add_argument("--target-glibc", default="2.32")
    p.add_argument("--quantize", choices=["none", "int8", "float16"], default="none")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-synthetic-validation", action="store_true")
    add_common(p); p.set_defaults(func=cmd_build)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except ModelToolError as error:
        print(json.dumps({"status": "error", "code": error.code, "message": str(error)}, indent=2),
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
