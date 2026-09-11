"""Portable, validated model-bundle input and delivery-package helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .errors import ModelToolError
from .manifest import read_json, write_json


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
    references: list[str]
    temporary_root: Path | None = None

    def cleanup(self) -> None:
        if self.temporary_root:
            shutil.rmtree(self.temporary_root, ignore_errors=True)

    def provenance(self) -> dict[str, Any]:
        assets: dict[str, str] = {"model": _sha256(self.model)}
        for label, file in (("metadata", self.metadata), ("model_profile", self.profile),
                            ("model_loader", self.loader)):
            if file:
                assets[label] = _sha256(file)
        for item in self.references:
            name, filename = item.split("=", 1)
            assets[f"reference:{name}"] = _sha256(Path(filename))
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


def load_bundle(value: str | Path) -> ModelBundle:
    root, temporary = _bundle_root(Path(value).expanduser())
    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        raise ModelToolError("BUNDLE_MANIFEST_MISSING", f"missing {manifest_path}")
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)
        raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"cannot read {manifest_path}: {error}") from error
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("bundle_id"), str):
        raise ModelToolError("BUNDLE_MANIFEST_INVALID", "bundle.json requires schema_version=1 and string bundle_id")

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
    references: list[str] = []
    raw_references = manifest.get("reference_inputs", [])
    if not isinstance(raw_references, list):
        raise ModelToolError("BUNDLE_MANIFEST_INVALID", "reference_inputs must be a list")
    seen_names: set[str] = set()
    for index, item in enumerate(raw_references):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("path"), str):
            raise ModelToolError("BUNDLE_MANIFEST_INVALID",
                                 f"reference_inputs[{index}] requires string name and path")
        name = item["name"]
        if not name or name in seen_names:
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"duplicate/empty reference name: {name!r}")
        seen_names.add(name)
        reference = _inside(root, root / item["path"], f"reference_inputs[{index}]")
        if reference.suffix.lower() != ".npy":
            raise ModelToolError("BUNDLE_MANIFEST_INVALID", f"reference input must be .npy: {reference}")
        references.append(f"{name}={reference}")

    return ModelBundle(root, manifest_path, manifest, model, optional_asset("metadata"),
                       optional_asset("model_profile"), optional_asset("model_loader"),
                       references, temporary)


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
