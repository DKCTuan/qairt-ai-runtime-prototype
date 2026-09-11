"""Portable, validated model-bundle input and delivery-package helpers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .errors import ModelToolError
from .manifest import read_json, write_json


_SUPPORTED_MODEL_SUFFIXES = {".tflite", ".onnx", ".pt", ".pth"}
_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, candidate: Path, label: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ModelToolError("BUNDLE_PATH_INVALID", f"{label} escapes bundle root: {candidate}") from error
    if not resolved.is_file():
        raise ModelToolError("BUNDLE_ASSET_MISSING", f"{label} not found: {candidate}")
    return resolved


def _validate_bundle_id(value: Any) -> str:
    if not isinstance(value, str) or not _BUNDLE_ID_RE.fullmatch(value):
        raise ModelToolError(
            "BUNDLE_MANIFEST_INVALID",
            "bundle_id must be 1-128 chars: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit",
        )
    return value


def _parse_reference_assets(root: Path, manifest: dict[str, Any], key: str) -> list[str]:
    references: list[str] = []
    raw_references = manifest.get(key, [])
    if not isinstance(raw_references, list):
        raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"{key} must be a list")
    seen_names: set[str] = set()
    for index, item in enumerate(raw_references):
        if (not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("path"), str)):
            raise ModelToolError("BUNDLE_MANIFEST_INVALID",
                                 f"{key}[{index}] requires string name and path")
        name = item["name"]
        if not name or name in seen_names:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"duplicate/empty {key} name: {name!r}")
        seen_names.add(name)
        reference = _inside(root, root / item["path"], f"{key}[{index}]")
        if reference.suffix.lower() != ".npy":
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"{key} must be .npy: {reference}")
        references.append(f"{name}={reference}")
    return references


def _safe_extract(zip_path: Path, destination: Path) -> None:
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination.resolve())
            except ValueError as error:
                raise ModelToolError("BUNDLE_ARCHIVE_INVALID",
                                     f"archive entry escapes destination: {member.filename}") from error
        archive.extractall(destination)


@dataclass
class ModelBundle:
    """A validated bundle; call ``cleanup`` after commands using a ZIP input."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    model: Path
    metadata: Path | None
    profile: Path | None
    loader: Path | None
    reference_inputs: list[str]
    reference_outputs: list[str]
    temporary_root: Path | None = None

    @property
    def references(self) -> list[str]:
        return self.reference_inputs

    def cleanup(self) -> None:
        if self.temporary_root:
            shutil.rmtree(self.temporary_root, ignore_errors=True)

    def provenance(self) -> dict[str, Any]:
        assets: dict[str, str] = {"model": _sha256(self.model)}
        for label, file in (("metadata", self.metadata), ("model_profile", self.profile),
                            ("model_loader", self.loader)):
            if file:
                assets[label] = _sha256(file)
        for item in self.reference_inputs:
            name, filename = item.split("=", 1)
            assets[f"reference_input:{name}"] = _sha256(Path(filename))
        for item in self.reference_outputs:
            name, filename = item.split("=", 1)
            assets[f"reference_output:{name}"] = _sha256(Path(filename))
        return {
            "schema_version": 1,
            "bundle_id": self.manifest["bundle_id"],
            "bundle_manifest": str(self.manifest_path),
            "assets_sha256": assets,
        }


def _bundle_root(input_path: Path) -> tuple[Path, Path | None]:
    if input_path.is_dir():
        return input_path.resolve(), None
    if input_path.suffix.lower() != ".zip" or not input_path.is_file():
        raise ModelToolError("BUNDLE_NOT_FOUND", "bundle must be a directory or .zip file")
    temporary = Path(tempfile.mkdtemp(prefix="modeltool-bundle-"))
    _safe_extract(input_path, temporary)
    if (temporary / "bundle.json").is_file():
        return temporary, temporary
    children = [item for item in temporary.iterdir() if item.is_dir()]
    if len(children) == 1 and (children[0] / "bundle.json").is_file():
        return children[0], temporary
    shutil.rmtree(temporary, ignore_errors=True)
    raise ModelToolError("BUNDLE_MANIFEST_MISSING", "bundle ZIP must contain bundle.json at its root")


def _validate_reference_tensors(*, np: Any, references: list[str],
                                specs: list[dict[str, Any]], label: str) -> list[str]:
    by_name = dict(item.split("=", 1) for item in references)
    expected_names = [item["name"] for item in specs]
    if set(by_name) != set(expected_names):
        raise ModelToolError("BUNDLE_REFERENCE_CONTRACT_MISMATCH",
                             f"{label} names differ: expected={expected_names}, got={sorted(by_name)}")
    for item in specs:
        try:
            tensor = np.load(by_name[item["name"]], allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ModelToolError("BUNDLE_REFERENCE_INVALID",
                                 f"cannot read {label} {item['name']!r}: {error}") from error
        actual_dtype, actual_shape = np.dtype(tensor.dtype).name, list(tensor.shape)
        if actual_dtype != item["dtype"] or actual_shape != item["shape"]:
            raise ModelToolError(
                "BUNDLE_REFERENCE_CONTRACT_MISMATCH",
                f"{label} {item['name']!r}: expected dtype={item['dtype']} shape={item['shape']}, "
                f"got dtype={actual_dtype} shape={actual_shape}",
            )
        if np.issubdtype(tensor.dtype, np.floating) and not np.isfinite(tensor).all():
            raise ModelToolError("BUNDLE_REFERENCE_INVALID",
                                 f"{label} {item['name']!r} contains NaN or Inf")
    return expected_names


def load_bundle(value: str | Path) -> ModelBundle:
    root, temporary = _bundle_root(Path(value).expanduser())
    try:
        manifest_path = root / "bundle.json"
        if not manifest_path.is_file():
            raise ModelToolError("BUNDLE_MANIFEST_MISSING", f"missing {manifest_path}")
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"cannot read {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", "bundle.json must contain a JSON object")
        if manifest.get("schema_version") != 1:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", "bundle.json requires schema_version=1")
        _validate_bundle_id(manifest.get("bundle_id"))

        def optional_asset(key: str) -> Path | None:
            value = manifest.get(key)
            if value is None:
                return None
            if not isinstance(value, str) or not value:
                raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"{key} must be a non-empty relative path")
            return _inside(root, root / value, key)

        model = optional_asset("model")
        if model is None:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", "bundle.json requires model")
        if model.suffix.lower() not in _SUPPORTED_MODEL_SUFFIXES:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID",
                                 "model must end in one of: " + ", ".join(sorted(_SUPPORTED_MODEL_SUFFIXES)))
        metadata = optional_asset("metadata")
        if metadata:
            try:
                metadata_value = read_json(metadata)
            except (OSError, json.JSONDecodeError) as error:
                raise ModelToolError("BUNDLE_METADATA_INVALID", f"cannot read metadata {metadata}: {error}") from error
            if not isinstance(metadata_value, dict):
                raise ModelToolError("BUNDLE_METADATA_INVALID", "metadata must contain a JSON object")
        loader = optional_asset("model_loader")
        if loader and model.suffix.lower() not in {".pt", ".pth"}:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", "model_loader is only valid for .pt/.pth models")
        reference_inputs = _parse_reference_assets(root, manifest, "reference_inputs")
        reference_outputs = _parse_reference_assets(root, manifest, "reference_outputs")

        return ModelBundle(root, manifest_path, manifest, model, metadata,
                           optional_asset("model_profile"), loader,
                           reference_inputs, reference_outputs, temporary)
    except Exception:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_bundle_contract(bundle: ModelBundle) -> dict[str, Any]:
    """Validate profile structure and real reference tensors before conversion.

    This deliberately checks only facts that can be established locally. Model
    graph operators are inspected by the framework adapter later in the build.
    """
    if bundle.profile is None:
        return {"status": "NOT_APPLICABLE", "reason": "bundle has no model_profile"}
    try:
        profile = read_json(bundle.profile)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelToolError("BUNDLE_PROFILE_INVALID",
                             f"cannot read model profile {bundle.profile}: {error}") from error
    if profile.get("schema_version") != 1:
        raise ModelToolError("BUNDLE_PROFILE_INVALID", "model_profile schema_version must be 1")
    inputs, outputs = profile.get("inputs"), profile.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or not outputs:
        raise ModelToolError("BUNDLE_PROFILE_INVALID", "model_profile requires non-empty inputs and outputs")
    supported_dtypes = {"float32", "int32"}
    seen_names: set[str] = set()
    for index, item in enumerate(inputs + outputs):
        group = "inputs" if index < len(inputs) else "outputs"
        local_index = index if group == "inputs" else index - len(inputs)
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise ModelToolError("BUNDLE_PROFILE_INVALID", f"{group}[{local_index}] needs a name")
        if item["name"] in seen_names:
            raise ModelToolError("BUNDLE_PROFILE_INVALID", f"duplicate tensor name: {item['name']}")
        seen_names.add(item["name"])
        if item.get("dtype") not in supported_dtypes:
            raise ModelToolError("BUNDLE_PROFILE_INVALID",
                                 f"{group}[{local_index}] has unsupported dtype {item.get('dtype')!r}")
        shape = item.get("shape")
        if (not isinstance(shape, list) or not shape
                or not all(type(size) is int and size > 0 for size in shape)):
            raise ModelToolError("BUNDLE_PROFILE_INVALID", f"{group}[{local_index}] needs a static positive shape")

    if not bundle.references:
        return {
            "status": "WARNING",
            "profile": str(bundle.profile),
            "reference_inputs": "not supplied",
            "message": "No golden input tensors: framework/TFLite parity cannot be representative. "
                       "Use --allow-synthetic-validation only for smoke tests.",
        }
    try:
        import numpy as np
    except ImportError as error:
        raise ModelToolError("BUNDLE_REFERENCE_INVALID", "numpy is required to validate .npy references") from error
    input_names = _validate_reference_tensors(np=np, references=bundle.reference_inputs,
                                              specs=inputs, label="reference input")
    result = {"status": "PASS", "profile": str(bundle.profile), "reference_inputs": input_names}
    if bundle.reference_outputs:
        result["reference_outputs"] = _validate_reference_tensors(
            np=np, references=bundle.reference_outputs, specs=outputs, label="reference output")
    else:
        result["reference_outputs"] = "not supplied"
    return result


def create_delivery_package(*, bundle: ModelBundle, build_root: Path, output: Path) -> Path:
    """Package deployable build outputs, profile and provenance; omit source checkpoints."""
    output = output.resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise ModelToolError("INVALID_OUTPUT", "--package-output must end in .tar.gz")
    if output.exists():
        raise ModelToolError("OUTPUT_EXISTS", f"package exists: {output}; use --force to replace it")
    delivery_name = output.name.removesuffix(".tar.gz")
    staging = Path(tempfile.mkdtemp(prefix="modeltool-delivery-")) / delivery_name
    try:
        (staging / "model").mkdir(parents=True)
        (staging / "include").mkdir()
        (staging / "lib").mkdir()
        (staging / "reports").mkdir()
        shutil.copy2(build_root / "converted" / "model_float32.tflite", staging / "model")
        for item in (build_root / "arm64").iterdir():
            if item.is_file():
                destination = (staging / "lib" if item.suffix in {".so", ".a"}
                               else staging / "include" if item.suffix == ".h"
                               else staging / "reports")
                shutil.copy2(item, destination)
        for name in ("manifest.json", "validation_report.json"):
            shutil.copy2(build_root / name, staging / "reports" / name)
        if bundle.profile:
            shutil.copy2(bundle.profile, staging / "model" / "model_profile.json")
        if bundle.metadata:
            shutil.copy2(bundle.metadata, staging / "model" / bundle.metadata.name)
        provenance = bundle.provenance()
        provenance["build_artifact_dir"] = str(build_root)
        write_json(staging / "BUNDLE_PROVENANCE.json", provenance)
        checksums = []
        for file in sorted(item for item in staging.rglob("*") if item.is_file()):
            checksums.append(f"{_sha256(file)}  {file.relative_to(staging)}")
        (staging / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging, arcname=delivery_name)
    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)
    return output
