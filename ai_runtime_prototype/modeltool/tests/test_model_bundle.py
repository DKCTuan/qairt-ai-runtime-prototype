import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modeltool.core.bundle import create_delivery_package, load_bundle, validate_bundle_contract
from modeltool.core.errors import ModelToolError


class ModelBundleTest(unittest.TestCase):
    def write_bundle(self, root: Path) -> None:
        (root / "model").mkdir()
        (root / "reference").mkdir()
        (root / "model" / "example.tflite").write_bytes(b"tflite")
        (root / "reference" / "input.npy").write_bytes(b"npy")
        (root / "bundle.json").write_text(json.dumps({
            "schema_version": 1,
            "bundle_id": "example-v1",
            "model": "model/example.tflite",
            "reference_inputs": [{"name": "input", "path": "reference/input.npy"}],
        }), encoding="utf-8")

    def test_directory_bundle_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            bundle = load_bundle(root)
            self.assertEqual("example-v1", bundle.manifest["bundle_id"])
            self.assertEqual([f"input={root / 'reference' / 'input.npy'}"], bundle.reference_inputs)
            self.assertEqual([], bundle.reference_outputs)
            self.assertEqual(64, len(bundle.provenance()["assets_sha256"]["model"]))

    def test_missing_references_are_warning_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            (root / "contract").mkdir()
            (root / "contract" / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "inputs": [{"name": "input", "dtype": "float32", "shape": [1, 2]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 1]}],
            }), encoding="utf-8")
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            manifest["model_profile"] = "contract/profile.json"
            manifest.pop("reference_inputs")
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual("WARNING", validate_bundle_contract(load_bundle(root))["status"])

    def test_zip_bundle_with_wrapper_directory_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wrapped"
            root.mkdir()
            self.write_bundle(root)
            archive = Path(directory) / "bundle.zip"
            with ZipFile(archive, "w") as output:
                for file in root.rglob("*"):
                    if file.is_file():
                        output.write(file, file.relative_to(root.parent))
            bundle = load_bundle(archive)
            try:
                self.assertEqual("example-v1", bundle.manifest["bundle_id"])
            finally:
                bundle.cleanup()

    def test_assets_cannot_escape_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bundle.json").write_text(json.dumps({
                "schema_version": 1, "bundle_id": "bad", "model": "../outside.tflite",
            }), encoding="utf-8")
            with self.assertRaises(ModelToolError) as context:
                load_bundle(root)
            self.assertEqual("BUNDLE_PATH_INVALID", context.exception.code)

    def test_bundle_id_must_be_filename_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            manifest["bundle_id"] = "../bad"
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ModelToolError) as context:
                load_bundle(root)
            self.assertEqual("BUNDLE_MANIFEST_INVALID", context.exception.code)

    def test_reference_tensors_must_match_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            (root / "contract").mkdir()
            (root / "contract" / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "inputs": [{"name": "input", "dtype": "float32", "shape": [1, 2]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 1]}],
            }), encoding="utf-8")
            np.save(root / "reference" / "input.npy", np.zeros((1, 2), dtype=np.float32))
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            manifest["model_profile"] = "contract/profile.json"
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
            bundle = load_bundle(root)
            self.assertEqual("PASS", validate_bundle_contract(bundle)["status"])
            np.save(root / "reference" / "input.npy", np.zeros((1, 3), dtype=np.float32))
            with self.assertRaises(ModelToolError) as context:
                validate_bundle_contract(bundle)
            self.assertEqual("BUNDLE_REFERENCE_CONTRACT_MISMATCH", context.exception.code)

    def test_reference_outputs_are_validated_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            (root / "contract").mkdir()
            (root / "contract" / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "inputs": [{"name": "input", "dtype": "float32", "shape": [1, 2]}],
                "outputs": [{"name": "scores", "dtype": "float32", "shape": [1, 5]}],
            }), encoding="utf-8")
            np.save(root / "reference" / "input.npy", np.zeros((1, 2), dtype=np.float32))
            np.save(root / "reference" / "scores.npy", np.zeros((1, 5), dtype=np.float32))
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            manifest["model_profile"] = "contract/profile.json"
            manifest["reference_outputs"] = [{"name": "scores", "path": "reference/scores.npy"}]
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
            bundle = load_bundle(root)
            report = validate_bundle_contract(bundle)
            self.assertEqual("PASS", report["status"])
            self.assertEqual(["scores"], report["reference_outputs"])
            self.assertIn("reference_output:scores", bundle.provenance()["assets_sha256"])

            np.save(root / "reference" / "scores.npy", np.zeros((1, 4), dtype=np.float32))
            with self.assertRaises(ModelToolError) as context:
                validate_bundle_contract(bundle)
            self.assertEqual("BUNDLE_REFERENCE_CONTRACT_MISMATCH", context.exception.code)

    def test_float_reference_tensors_must_be_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            (root / "contract").mkdir()
            (root / "contract" / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "inputs": [{"name": "input", "dtype": "float32", "shape": [1, 2]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 1]}],
            }), encoding="utf-8")
            np.save(root / "reference" / "input.npy", np.array([[0.0, np.nan]], dtype=np.float32))
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            manifest["model_profile"] = "contract/profile.json"
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ModelToolError) as context:
                validate_bundle_contract(load_bundle(root))
            self.assertEqual("BUNDLE_REFERENCE_INVALID", context.exception.code)

    def test_profile_rejects_empty_names_and_bool_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_bundle(root)
            (root / "contract").mkdir()
            manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
            manifest["model_profile"] = "contract/profile.json"
            manifest.pop("reference_inputs")
            (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")

            (root / "contract" / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "inputs": [{"name": "", "dtype": "float32", "shape": [1, 2]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 1]}],
            }), encoding="utf-8")
            with self.assertRaises(ModelToolError) as context:
                validate_bundle_contract(load_bundle(root))
            self.assertEqual("BUNDLE_PROFILE_INVALID", context.exception.code)

            (root / "contract" / "profile.json").write_text(json.dumps({
                "schema_version": 1,
                "inputs": [{"name": "input", "dtype": "float32", "shape": [True, 2]}],
                "outputs": [{"name": "output", "dtype": "float32", "shape": [1, 1]}],
            }), encoding="utf-8")
            with self.assertRaises(ModelToolError) as context:
                validate_bundle_contract(load_bundle(root))
            self.assertEqual("BUNDLE_PROFILE_INVALID", context.exception.code)

    def test_delivery_package_contains_headers_and_not_source_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            root.mkdir()
            self.write_bundle(root)
            bundle = load_bundle(root)
            build = Path(directory) / "build"
            (build / "converted").mkdir(parents=True)
            (build / "arm64").mkdir()
            (build / "converted" / "model_float32.tflite").write_bytes(b"converted")
            (build / "arm64" / "libexample.so").write_bytes(b"library")
            (build / "arm64" / "ai_model.h").write_text("/* header */\n", encoding="utf-8")
            (build / "manifest.json").write_text("{}\n", encoding="utf-8")
            (build / "validation_report.json").write_text("{}\n", encoding="utf-8")
            archive = create_delivery_package(bundle=bundle, build_root=build,
                                              output=Path(directory) / "delivery.tar.gz")
            with tarfile.open(archive, "r:gz") as contents:
                names = contents.getnames()
            self.assertTrue(any(name.endswith("/include/ai_model.h") for name in names))
            self.assertTrue(any(name.endswith("/lib/libexample.so") for name in names))
            self.assertFalse(any(name.endswith("example.tflite") and "/source/" in name for name in names))


if __name__ == "__main__":
    unittest.main()
