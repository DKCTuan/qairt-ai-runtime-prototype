import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_module("pytorch_to_tflite", ROOT / "tools" / "pytorch_to_tflite.py")
sys.path.insert(0, str(ROOT))
from modeltool.core.errors import ModelToolError
from modeltool.modeltool import validate_tflite_profile


class MultiInputContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile_path = self.root / "model_profile.json"
        self.profile_path.write_text(json.dumps({
            "schema_version": 1,
            "inputs": [
                {"name": "numeric", "dtype": "float32", "shape": [1, 90, 3]},
                {"name": "p16_ids", "dtype": "int32", "shape": [1, 90]},
            ],
        }), encoding="utf-8")
        self.numeric = self.root / "numeric.npy"
        self.ids = self.root / "ids.npy"
        self.numeric.touch()
        self.ids.touch()

    def tearDown(self):
        self.temporary.cleanup()

    def test_named_references_follow_profile_order(self):
        path, _, specs = exporter.load_profile(str(self.profile_path))
        self.assertEqual(path, self.profile_path)
        self.assertEqual([item["dtype"] for item in specs], ["float32", "int32"])
        paths = exporter.reference_paths([
            f"p16_ids={self.ids}", f"numeric={self.numeric}",
        ], specs)
        self.assertEqual(paths, [self.numeric, self.ids])

    def test_positional_reference_count_is_checked(self):
        _, _, specs = exporter.load_profile(str(self.profile_path))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                exporter.reference_paths([str(self.numeric)], specs)

    def test_tflite_contract_mismatch_is_rejected(self):
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        profile["outputs"] = [{"name": "scores", "dtype": "float32", "shape": [1, 5]}]
        io = {
            "inputs": [
                {"name": "numeric", "dtype": "float32", "shape": [1, 90, 3]},
                {"name": "p16_ids", "dtype": "int64", "shape": [1, 90]},
            ],
            "outputs": [{"name": "scores", "dtype": "float32", "shape": [1, 5]}],
        }
        with self.assertRaises(ModelToolError) as context:
            validate_tflite_profile(io, profile)
        self.assertEqual(context.exception.code, "MODEL_PROFILE_MISMATCH")


if __name__ == "__main__":
    unittest.main()
