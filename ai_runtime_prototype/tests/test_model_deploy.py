import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'model_deploy.py'
SPEC = importlib.util.spec_from_file_location('model_deploy', MODULE_PATH)
model_deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_deploy)


class ModelDeployTests(unittest.TestCase):
    def test_static_library_quantize_option_defaults_to_none(self):
        parser = model_deploy.build_parser()
        defaults = parser.parse_args([
            'static-library', '--model', 'model.tflite', '--output', 'libai_model.so',
        ])
        int8 = parser.parse_args([
            'static-library', '--model', 'model.tflite', '--output', 'libai_model.so',
            '--quantize', 'int8',
        ])
        self.assertEqual(defaults.quantize, 'none')
        self.assertEqual(int8.quantize, 'int8')

    def test_quantized_tensor_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.raw'
            path.write_bytes(struct.pack('<3b', -2, 0, 4))
            self.assertEqual(model_deploy.read_tensor_values(path, 'int8', 0.5, 0),
                             [-1.0, 0.0, 2.0])

    def test_htp_bundle_contains_checksums_and_matching_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / 'sdk'
            target = 'aarch64-oe-linux-gcc11.2'
            target_lib = sdk / 'lib' / target
            skel_lib = sdk / 'lib' / 'hexagon-v73' / 'unsigned'
            target_lib.mkdir(parents=True)
            skel_lib.mkdir(parents=True)
            files = {
                target_lib / 'libQnnHtp.so': b'backend',
                target_lib / 'libQnnSystem.so': b'system',
                target_lib / 'libQnnHtpV73Stub.so': b'stub',
                skel_lib / 'libQnnHtpV73Skel.so': b'skel',
                root / 'model.bin': b'context',
                root / 'app': b'executable',
            }
            for path, data in files.items():
                path.write_bytes(data)

            output = root / 'bundle'
            args = argparse.Namespace(
                qairt_root=str(sdk), artifact=str(root / 'model.bin'),
                output=str(output), target=target, backend='htp', htp_arch='v73',
                app=str(root / 'app'), artifact_kind='context-binary',
            )
            with contextlib.redirect_stdout(io.StringIO()):
                model_deploy.command_package_target(args)

            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['htp']['architecture'], 'V73')
            self.assertEqual(set(manifest['sha256']), {
                'model/model.bin', 'lib/libQnnHtp.so', 'lib/libQnnSystem.so',
                'app/app', 'lib/libQnnHtpV73Stub.so', 'skel/libQnnHtpV73Skel.so',
            })
            for relative, expected in manifest['sha256'].items():
                actual = hashlib.sha256((output / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)
            self.assertIn('ADSP_LIBRARY_PATH', (output / 'run.sh').read_text())


if __name__ == '__main__':
    unittest.main()
