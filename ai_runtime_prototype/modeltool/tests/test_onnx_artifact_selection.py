import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_runtime_prototype.modeltool.adapters.onnx_adapter import convert_onnx
from ai_runtime_prototype.modeltool.core.errors import ModelToolError


class ArtifactSelectionTests(unittest.TestCase):
    def run_conversion(self, names):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            converter = root / "converter"
            converter.touch()
            output = root / "model.tflite"
            output.write_bytes(b"existing")

            def fake_run(command, **kwargs):
                work = Path(command[command.index("-o") + 1])
                for name in names:
                    (work / name).write_bytes(name.encode())

            with patch("ai_runtime_prototype.modeltool.adapters.onnx_adapter.run", fake_run):
                try:
                    convert_onnx(root / "source.onnx", output, converter=converter,
                                 input_shapes=[], log_path=root / "log")
                except ModelToolError:
                    self.assertEqual(output.read_bytes(), b"existing")
                    raise
            return output.read_bytes()

    def test_selects_float32_not_first_file(self):
        self.assertEqual(self.run_conversion(["a_float16.tflite", "z_float32.tflite"]),
                         b"z_float32.tflite")

    def test_rejects_missing_float32_without_overwrite(self):
        with self.assertRaises(ModelToolError):
            self.run_conversion(["model_float16.tflite"])

    def test_rejects_ambiguous_float32_without_overwrite(self):
        with self.assertRaises(ModelToolError):
            self.run_conversion(["a_float32.tflite", "b_float32.tflite"])
