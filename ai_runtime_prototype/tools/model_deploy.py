#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_WORK_ROOT = Path('/tmp/model_deploy_tool')

# Target triplet mac dinh = host x86_64. Cac lenh convert/run-host/validate
# (da co tu truoc) luon chay tren host nen khong doi. build-context/run-api/
# deploy (moi) nhan them --target de tro toi thu muc bin/lib khac trong SDK -
# vi du khi cross-compile runtime cho device nhung that (aarch64-...).
DEFAULT_TARGET = 'x86_64-linux-clang'

# Ten thu vien backend QNN theo loai phan cung. build-context dung de chon
# .so nao duoc "bake" vao context binary; run-api dung de chon .so nao
# traffic_app se dlopen luc chay. LUU Y: tren HTP, qnn-context-binary-generator
# thuong van chay bang binary host (x86_64-linux-clang) nhung build context
# cho HTP thong qua thu vien "offline prepare" - hanh vi chinh xac phu thuoc
# SDK version, nen kiem tra lai voi `qnn-context-binary-generator --help`
# va tai lieu QAIRT cua ban truoc khi dung --backend htp/gpu that su.
BACKEND_LIBS = {
    'cpu': 'libQnnCpu.so',
    'gpu': 'libQnnGpu.so',
    'htp': 'libQnnHtp.so',
}


def htp_runtime_paths(qairt_root: Path, target: str, htp_arch: str):
    """Return the AP-side Stub and Hexagon-side Skel for one HTP architecture."""
    normalized = htp_arch.lower().removeprefix('v')
    if not normalized.isdigit():
        fail(f'invalid HTP architecture {htp_arch!r}; expected for example v73')
    arch = f'V{normalized}'
    return {
        'arch': arch,
        'stub': qairt_root / 'lib' / target / f'libQnnHtp{arch}Stub.so',
        'skel': qairt_root / 'lib' / f'hexagon-v{normalized}' / 'unsigned' / f'libQnnHtp{arch}Skel.so',
    }


def discover_qairt_root() -> Path:
    """Find the SDK without tying the tool to one user's home directory."""
    for name in ('DL_QAIRT_ROOT', 'QAIRT_ROOT', 'QNN_SDK_ROOT'):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()

    sdk_parent = Path(__file__).resolve().parents[2] / 'qairt'
    if sdk_parent.is_dir():
        candidates = sorted(
            (path for path in sdk_parent.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()

    return Path('/home/congtuan/qairt_sdk/qairt/2.44.0.260225')


DEFAULT_QAIRT_ROOT = discover_qairt_root()


def fail(message: str, code: int = 1):
    print(f'ERROR: {message}', file=sys.stderr)
    raise SystemExit(code)


def ensure_file(path: Path, label: str):
    if not path.is_file():
        fail(f'{label} not found: {path}')


def qairt_paths(qairt_root: Path, target: str = DEFAULT_TARGET):
    bin_dir = qairt_root / 'bin' / target
    lib_dir = qairt_root / 'lib' / target
    return {
        'bin': bin_dir,
        'lib': lib_dir,
        'qnn_net_run': bin_dir / 'qnn-net-run',
        'qnn_context_binary_generator': bin_dir / 'qnn-context-binary-generator',
        'qnn_cpu': lib_dir / 'libQnnCpu.so',
        'qnn_system': lib_dir / 'libQnnSystem.so',
        'qnn_model_dlc': lib_dir / 'libQnnModelDlc.so',
        'qairt_converter': bin_dir / 'qairt-converter',
        'qnn_onnx_converter': bin_dir / 'qnn-onnx-converter',
        'qnn_tflite_converter': bin_dir / 'qnn-tflite-converter',
        'qnn_pytorch_converter': bin_dir / 'qnn-pytorch-converter',
        'snpe_onnx_to_dlc': bin_dir / 'snpe-onnx-to-dlc',
        'snpe_tflite_to_dlc': bin_dir / 'snpe-tflite-to-dlc',
    }


def backend_lib_path(qairt_root: Path, target: str, backend: str):
    try:
        lib_name = BACKEND_LIBS[backend]
    except KeyError:
        fail(f'unsupported backend: {backend}; choose one of {", ".join(BACKEND_LIBS)}')
    return qairt_root / 'lib' / target / lib_name


def run(cmd, env=None, cwd=None, log_path: Path | None = None):
    printable = ' '.join(str(x) for x in cmd)
    print(f'$ {printable}')
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('w') as log:
                proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        else:
            proc = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    except OSError as exc:
        fail(f'cannot execute {cmd[0]}: {exc}')
    if proc.returncode != 0:
        if log_path is not None:
            fail(f'command failed, see log: {log_path}')
        fail('command failed')


def run_capture(cmd, env=None, cwd=None):
    """Giong run() nhung capture stdout/stderr thay vi in truc tiep - can
    cho run-api de parse 'label=...'/'scores=...' tu traffic_app."""
    printable = ' '.join(str(x) for x in cmd)
    print(f'$ {printable}')
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    except OSError as exc:
        fail(f'cannot execute {cmd[0]}: {exc}')


def parse_traffic_app_output(text: str):
    """Parse stdout cua examples/main.c: 'label=N', 'scores=f,f,f,...' va
    'latency_ms=f'. So luong score la DONG (khong con co dinh CLASS_NUM=5)
    vi runtime API da doc shape that tu model - xem
    include/ai_runtime.h/dl_get_io_count. latency_ms co the la None neu
    chay voi binary cu (build truoc khi them do latency) - cac cho goi
    ham nay phai tu xu ly truong hop None, khong gia dinh luon co gia tri."""
    label = None
    scores = []
    latency_ms = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('label='):
            try:
                label = int(line[len('label='):])
            except ValueError:
                pass
        elif line.startswith('scores='):
            raw = line[len('scores='):]
            if raw:
                try:
                    scores = [float(x) for x in raw.split(',')]
                except ValueError:
                    scores = []
        elif line.startswith('latency_ms='):
            try:
                latency_ms = float(line[len('latency_ms='):])
            except ValueError:
                pass
    return label, scores, latency_ms


def result_with_limited_scores(result: dict, limit: int = 100):
    """Keep CLI/REST output bounded for detector models with huge tensors."""
    printable = dict(result)
    scores = printable.get('scores')
    if isinstance(scores, list):
        printable['score_count'] = len(scores)
        if limit >= 0 and len(scores) > limit:
            printable['scores'] = scores[:limit]
            printable['scores_truncated'] = True
        else:
            printable['scores_truncated'] = False
    return printable


def make_env(qairt_root: Path, target: str = DEFAULT_TARGET):
    paths = qairt_paths(qairt_root, target)
    env = os.environ.copy()
    env['PATH'] = f"{paths['bin']}:{env.get('PATH', '')}"
    env['LD_LIBRARY_PATH'] = f"{paths['lib']}:{env.get('LD_LIBRARY_PATH', '')}"
    python_dir = qairt_root / 'lib' / 'python'
    env['PYTHONPATH'] = f"{python_dir}:{env.get('PYTHONPATH', '')}"
    env['SNPE_ROOT'] = str(qairt_root)
    env['QNN_SDK_ROOT'] = str(qairt_root)
    env['QAIRT_ROOT'] = str(qairt_root)
    return env


def ensure_exists(path: Path, label: str):
    if not path.exists():
        fail(f'{label} not found: {path}')


def qairt_python(qairt_root: Path):
    bundled = qairt_root / 'qairt_env' / 'bin' / 'python'
    if bundled.is_file():
        return bundled
    current = Path(sys.executable).resolve()
    if current.is_file():
        return current
    fail(f'QAIRT Python environment not found below {qairt_root} and no current Python is usable')


def model_from_args(args) -> Path:
    value = getattr(args, 'model', None) or getattr(args, 'model_positional', None)
    if not value:
        fail('input model is required (use MODEL or --model MODEL)')
    return Path(value).expanduser().resolve()


def normalized_extra_args(values):
    values = list(values or [])
    return values[1:] if values[:1] == ['--'] else values


def command_doctor(args):
    """Read-only preflight before spending time converting or deploying a model."""
    qairt_root = Path(args.qairt_root).expanduser().resolve()
    paths = qairt_paths(qairt_root, args.target)
    host_paths = qairt_paths(qairt_root, DEFAULT_TARGET)
    checks = []

    def check(name, path, file_expected=True, required=True, hint=None):
        path = Path(path)
        ok = path.is_file() if file_expected else path.is_dir()
        item = {'name': name, 'path': str(path), 'required': required, 'ok': ok}
        if hint:
            item['hint'] = hint
        checks.append(item)

    check('QAIRT root', qairt_root, file_expected=False)
    check('target bin directory', paths['bin'], file_expected=False)
    check('target lib directory', paths['lib'], file_expected=False)
    check('libQnnSystem.so', paths['qnn_system'])
    check(f'libQnn{args.backend.capitalize()}.so',
          backend_lib_path(qairt_root, args.target, args.backend))

    if args.backend == 'htp' and args.htp_arch:
        htp_paths = htp_runtime_paths(qairt_root, args.target, args.htp_arch)
        check(f'HTP {htp_paths["arch"]} Stub', htp_paths['stub'])
        check(f'HTP {htp_paths["arch"]} Skel', htp_paths['skel'])

    if args.model:
        model = Path(args.model).expanduser().resolve()
        check('input model', model)
        extension = model.suffix.lower()
        if extension == '.onnx':
            check('host qairt-converter', host_paths['qairt_converter'])
            check('host qnn-onnx-converter', host_paths['qnn_onnx_converter'], required=False)
            check('host qnn-context-binary-generator', host_paths['qnn_context_binary_generator'])
        elif extension == '.tflite':
            check('host qairt-converter', host_paths['qairt_converter'])
            check('host qnn-tflite-converter', host_paths['qnn_tflite_converter'], required=False)
            check('host qnn-context-binary-generator', host_paths['qnn_context_binary_generator'])
        elif extension in ('.pt', '.pth'):
            check('host qnn-pytorch-converter', host_paths['qnn_pytorch_converter'])
            check('host qnn-context-binary-generator', host_paths['qnn_context_binary_generator'])
            is_torchscript = model.is_file() and is_torchscript_archive(model)
            checks.append({
                'name': 'TorchScript archive', 'path': str(model), 'required': True,
                'ok': is_torchscript,
                'hint': 'export a training checkpoint with torch.jit.trace/script before conversion',
            })
        elif extension == '.dlc':
            check('host qnn-context-binary-generator', host_paths['qnn_context_binary_generator'])
        elif extension == '.bin':
            checks.append({
                'name': 'context binary artifact', 'path': str(model), 'required': True,
                'ok': model.is_file(),
                'hint': 'binary compatibility with the board is validated only at target runtime',
            })
        else:
            checks.append({
                'name': 'supported model extension', 'path': extension or '<none>', 'required': True,
                'ok': False, 'hint': 'use ONNX, TFLite, TorchScript .pt/.pth, DLC, or a QNN context .bin',
            })

    missing = [item for item in checks if item['required'] and not item['ok']]
    warnings = []
    if args.target != DEFAULT_TARGET:
        warnings.append('This target can be prepared, but run-api executes only local x86_64 binaries. '
                        'Copy artifacts and run them on the matching device.')
    if args.backend == 'htp':
        if not args.htp_arch:
            warnings.append('Pass --htp-arch (for example v73) to also validate the matching Stub and Skel files.')
        warnings.append('HTP preflight cannot validate the target SoC, firmware, driver, or graph partition. '
                        'Validate those on the physical Qualcomm device.')
    result = {
        'status': 'ready' if not missing else 'not_ready',
        'qairt_root': str(qairt_root), 'target': args.target, 'backend': args.backend,
        'checks': checks, 'warnings': warnings,
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)
    return result


def copy_into_bundle(source: Path, destination_dir: Path, label: str):
    """Copy one required deployment file; no glob means no accidental payload."""
    ensure_file(source, label)
    if source.name in ('.', '..') or any(
            not (character.isalnum() or character in '._-') for character in source.name):
        fail(f'{label} filename contains shell-unsafe characters: {source.name!r}; rename it first')
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return destination


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command_package_target(args):
    """Create a reproducible QAIRT Linux bundle without connecting to a device.

    Qualcomm HTP needs application-processor libraries plus an architecture-
    matched Hexagon Skel. Keeping both in a manifest makes ABI/HTP mismatches
    visible before a user copies anything to a physical board.
    """
    qairt_root = Path(args.qairt_root).expanduser().resolve()
    artifact = Path(args.artifact).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ensure_file(artifact, 'deployment artifact')
    if args.artifact_kind != 'context-binary':
        fail('package-target currently packages the direct QNN API runtime and therefore requires '
             '--artifact-kind context-binary; DLC and TFLite need different entrypoints/libraries')
    if output.exists():
        fail(f'output bundle already exists: {output}; choose a new path to avoid overwriting it')

    paths = qairt_paths(qairt_root, args.target)
    app_dir, lib_dir = output / 'app', output / 'lib'
    model_dir, skel_dir = output / 'model', output / 'skel'
    copied_artifact = copy_into_bundle(artifact, model_dir, 'deployment artifact')
    copied_backend = copy_into_bundle(backend_lib_path(qairt_root, args.target, args.backend),
                                      lib_dir, f'{args.backend} backend library')
    copied_system = copy_into_bundle(paths['qnn_system'], lib_dir, 'QNN System library')
    copied_app = None
    if args.app:
        copied_app = copy_into_bundle(Path(args.app).expanduser().resolve(), app_dir, 'application binary')
        copied_app.chmod(copied_app.stat().st_mode | 0o111)

    htp = None
    if args.backend == 'htp':
        if not args.htp_arch:
            fail('--htp-arch (for example v73) is required when --backend htp')
        htp_paths = htp_runtime_paths(qairt_root, args.target, args.htp_arch)
        htp = {
            'architecture': htp_paths['arch'],
            'stub': str(copy_into_bundle(htp_paths['stub'], lib_dir, 'HTP Stub library').relative_to(output)),
            'skel': str(copy_into_bundle(htp_paths['skel'], skel_dir, 'HTP Skel library').relative_to(output)),
        }

    entrypoint = f'./app/{copied_app.name}' if copied_app else None
    run_script = output / 'run.sh'
    lines = [
        '#!/bin/sh', 'set -eu',
        'ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
        'export LD_LIBRARY_PATH="$ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
        'export DL_QNN_SYSTEM_LIB="$ROOT/lib/libQnnSystem.so"',
        f'export DL_QNN_BACKEND_LIB="$ROOT/lib/{copied_backend.name}"',
        f'export DL_MODEL_PATH="$ROOT/model/{copied_artifact.name}"',
    ]
    if htp:
        # QNN HTP/Hexagon lookup uses an ADSP search path.  Use ';' rather
        # than ':' because it is consumed by the DSP loader, not ld.so.
        lines.append('export ADSP_LIBRARY_PATH="$ROOT/skel${ADSP_LIBRARY_PATH:+;$ADSP_LIBRARY_PATH}"')
    if entrypoint:
        lines.append(f'exec "$ROOT/{entrypoint}" "$@"')
    else:
        lines.extend([
            'echo "Bundle has no application binary. Re-run package-target with --app <target-built binary>." >&2',
            'exit 64',
        ])
    run_script.write_text('\n'.join(lines) + '\n')
    run_script.chmod(0o755)

    payload_files = [copied_artifact, copied_backend, copied_system]
    if copied_app:
        payload_files.append(copied_app)
    if htp:
        payload_files.extend([output / htp['stub'], output / htp['skel']])
    manifest = {
        'format_version': 1,
        'artifact': {'path': str(copied_artifact.relative_to(output)), 'kind': args.artifact_kind},
        'target': args.target,
        'backend': args.backend,
        'application': str(copied_app.relative_to(output)) if copied_app else None,
        'libraries': [str(copied_backend.relative_to(output)), str(copied_system.relative_to(output))],
        'sha256': {str(path.relative_to(output)): sha256_file(path) for path in payload_files},
        'htp': htp,
        'run_script': 'run.sh',
        'notes': [
            'Copy the complete directory only to a board with the exact target ABI.',
            'For HTP, install the Skel only through the board/BSP-approved DSP filesystem path.',
            'Successful launch does not prove full graph delegation; inspect target logs and profiling.',
        ],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'success', 'bundle': str(output),
                      'manifest': str(output / 'manifest.json')}, indent=2))
    return manifest


def is_torchscript_archive(path: Path) -> bool:
    """TorchScript archives contain compiled graph code; state_dict checkpoints do not."""
    try:
        with zipfile.ZipFile(path) as archive:
            return any('/code/' in name or name.startswith('code/') for name in archive.namelist())
    except zipfile.BadZipFile:
        return False


def command_convert(args):
    qairt_root = Path(args.qairt_root).expanduser().resolve()
    paths = qairt_paths(qairt_root)
    model = model_from_args(args)
    ensure_file(model, 'input model')
    output_value = getattr(args, 'output', None)
    output = Path(output_value).expanduser().resolve() if output_value else model.with_suffix('.dlc')
    output.parent.mkdir(parents=True, exist_ok=True)

    ext = model.suffix.lower()

    if ext == '.dlc':
        if model != output:
            shutil.copy2(model, output)
            mode = 'copy'
        else:
            mode = 'reuse'
        result = {'status': 'success', 'dlc_path': str(output), 'mode': mode}
        print(json.dumps(result, indent=2))
        return result

    if ext not in ('.onnx', '.tflite', '.pt', '.pth'):
        fail(f'unsupported input model extension: {ext or "<none>"}; expected .onnx, .tflite, '
             '.pt/.pth TorchScript, or .dlc')

    # Chi resolve QAIRT python env khi thuc su can converter (input khong
    # phai .dlc) - truoc day goi ham nay vo dieu kien nen lenh convert/deploy
    # fail oan neu SDK thieu qairt_env/, ke ca khi input da la .dlc va khong
    # can Python converter nao ca.
    env = make_env(qairt_root)
    python = qairt_python(qairt_root)
    converter = args.converter

    if converter == 'auto':
        converter = 'qnn' if ext in ('.pt', '.pth') else 'qairt'

    if converter == 'qairt':
        if ext in ('.pt', '.pth'):
            fail('qairt-converter does not accept a PyTorch checkpoint here; use --converter qnn '
                 'with a TorchScript model, or export the model to ONNX first')
        ensure_file(paths['qairt_converter'], 'qairt-converter')
        cmd = [python, paths['qairt_converter'], '--input_network', model, '--output_path', output]
    elif converter == 'snpe':
        if ext == '.onnx':
            ensure_file(paths['snpe_onnx_to_dlc'], 'snpe-onnx-to-dlc')
            cmd = [python, paths['snpe_onnx_to_dlc'], '--input_network', model, '--output_path', output]
        elif ext == '.tflite':
            ensure_file(paths['snpe_tflite_to_dlc'], 'snpe-tflite-to-dlc')
            cmd = [python, paths['snpe_tflite_to_dlc'], '--input_network', model, '--output_path', output]
        else:
            fail(f'snpe converter only supports .onnx/.tflite in this prototype, got: {ext}')
    elif converter == 'qnn':
        if ext == '.onnx':
            ensure_file(paths['qnn_onnx_converter'], 'qnn-onnx-converter')
            cmd = [python, paths['qnn_onnx_converter'], '--input_network', model, '--output_path', output]
        elif ext == '.tflite':
            ensure_file(paths['qnn_tflite_converter'], 'qnn-tflite-converter')
            cmd = [python, paths['qnn_tflite_converter'], '--input_network', model, '--output_path', output]
        elif ext in ('.pt', '.pth'):
            if not is_torchscript_archive(model):
                fail('PyTorch file is not a TorchScript archive. A training checkpoint/state_dict cannot '
                     'be converted by itself: load it with the original model class, then export via '
                     'torch.jit.trace/script or torch.onnx.export. See README: PyTorch export.')
            ensure_file(paths['qnn_pytorch_converter'], 'qnn-pytorch-converter')
            input_dims = getattr(args, 'pytorch_input_dim', None) or []
            if not input_dims:
                fail('PyTorch conversion requires --pytorch-input-dim INPUT_NAME DIMS, for example '
                     '--pytorch-input-dim input 1,80,3')
            cmd = [python, paths['qnn_pytorch_converter'], '--input_network', model, '--output_path', output]
            for item in input_dims:
                cmd.extend(['--input_dim', item[0], item[1]])
        else:
            fail(f'qnn converter only supports .onnx/.tflite in this prototype, got: {ext}')
    else:
        fail(f'unsupported converter: {converter}')

    if args.source_model_input_shape:
        for item in args.source_model_input_shape:
            if ext not in ('.pt', '.pth'):
                cmd.extend(['--source_model_input_shape', item[0], item[1]])

    if args.out_tensor_node:
        for name in args.out_tensor_node:
            cmd.extend(['--out_tensor_node', name])

    cmd.extend(normalized_extra_args(getattr(args, 'extra_args', [])))

    log_path = output.with_suffix('.convert.log')
    run(cmd, env=env, log_path=log_path)
    result = {
        'status': 'success',
        'converter': converter,
        'dlc_path': str(output),
        'log_path': str(log_path),
    }
    print(json.dumps(result, indent=2))
    return result


def write_input_list(input_name: str, input_raw: Path, input_list: Path):
    input_list.write_text(f'{input_name}:={input_raw}\n')


def read_float32(path: Path):
    data = path.read_bytes()
    if len(data) % 4 != 0:
        fail(f'raw output size is not float32-aligned: {path}')
    return list(struct.unpack('<' + 'f' * (len(data) // 4), data))


def read_tensor_values(path: Path, dtype='float32', scale=1.0, zero_point=0):
    data = path.read_bytes()
    if dtype == 'float32':
        return read_float32(path)
    if scale <= 0.0:
        fail(f'quantized output scale must be positive, got {scale}')
    if dtype == 'uint8':
        raw = data
    elif dtype == 'int8':
        raw = struct.unpack(f'<{len(data)}b', data)
    else:
        fail(f'unsupported tensor dtype: {dtype}')
    return [scale * (float(value) - float(zero_point)) for value in raw]


def command_run_host(args):
    qairt_root = Path(args.qairt_root)
    paths = qairt_paths(qairt_root)
    dlc = Path(args.dlc).resolve()
    input_raw = Path(args.input_raw).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    ensure_file(paths['qnn_net_run'], 'qnn-net-run')
    ensure_file(paths['qnn_cpu'], 'libQnnCpu.so')
    ensure_file(paths['qnn_model_dlc'], 'libQnnModelDlc.so')
    ensure_file(dlc, 'DLC model')
    ensure_file(input_raw, 'input raw')

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_list = work_dir / 'input_list.txt'
    write_input_list(args.input_name, input_raw, input_list)

    env = make_env(qairt_root)
    log_path = output_dir / 'qnn-net-run.log'
    cmd = [
        paths['qnn_net_run'],
        '--backend', paths['qnn_cpu'],
        '--model', paths['qnn_model_dlc'],
        '--dlc_path', dlc,
        '--input_list', input_list,
        '--output_dir', output_dir,
        '--log_level', args.log_level,
    ]
    run(cmd, env=env, log_path=log_path)

    output_raw = output_dir / 'Result_0' / f'{args.output_name}.raw'
    ensure_file(output_raw, 'QNN output')
    scores = read_float32(output_raw)
    label = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
    result = {'status': 'success', 'label': label, 'scores': scores, 'output_raw': str(output_raw), 'log_path': str(log_path)}
    print(json.dumps(result_with_limited_scores(
        result, getattr(args, 'max_print_scores', 100)), indent=2))
    return result


def read_npy_float32(path: Path):
    try:
        import numpy as np
    except Exception as exc:
        fail(f'numpy is required to read .npy reference: {exc}')
    return [float(x) for x in np.load(path).reshape(-1).astype('float32')]


def command_validate(args):
    command_run_host(args)
    output_raw = Path(args.output_dir).resolve() / 'Result_0' / f'{args.output_name}.raw'
    qnn = read_tensor_values(output_raw, args.output_dtype, args.output_scale,
                             args.output_zero_point)
    ref_path = Path(args.reference).resolve()
    ensure_file(ref_path, 'reference output')

    if ref_path.suffix.lower() == '.npy':
        ref_values = read_npy_float32(ref_path)
    elif ref_path.suffix.lower() == '.raw':
        ref_values = read_tensor_values(ref_path, args.reference_dtype,
                                        args.reference_scale, args.reference_zero_point)
    else:
        fail('reference must be .npy or .raw')

    n = min(len(qnn), len(ref_values))
    diffs = [abs(qnn[i] - ref_values[i]) for i in range(n)]
    max_abs_diff = max(diffs) if diffs else float('inf')
    mean_abs_diff = sum(diffs) / len(diffs) if diffs else float('inf')
    qnn_argmax = max(range(len(qnn)), key=lambda i: qnn[i]) if qnn else -1
    ref_argmax = max(range(len(ref_values)), key=lambda i: ref_values[i]) if ref_values else -1
    relative_failures = [
        i for i in range(n)
        if diffs[i] > args.tolerance + args.relative_tolerance * abs(ref_values[i])
    ]
    dot = sum(qnn[i] * ref_values[i] for i in range(n))
    qnn_norm = sum(qnn[i] * qnn[i] for i in range(n)) ** 0.5
    ref_norm = sum(ref_values[i] * ref_values[i] for i in range(n)) ** 0.5
    cosine_similarity = dot / (qnn_norm * ref_norm) if qnn_norm > 0.0 and ref_norm > 0.0 else None
    passed = len(qnn) == len(ref_values) and not relative_failures
    if args.require_argmax:
        passed = passed and qnn_argmax == ref_argmax

    result = {
        'status': 'success' if passed else 'failed',
        'validation_passed': passed,
        'qnn_shape': [len(qnn)],
        'reference_shape': [len(ref_values)],
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'cosine_similarity': cosine_similarity,
        'mismatch_count': len(relative_failures),
        'qnn_argmax': qnn_argmax,
        'reference_argmax': ref_argmax,
        'tolerance': args.tolerance,
        'relative_tolerance': args.relative_tolerance,
        'output_dtype': args.output_dtype,
        'reference_dtype': args.reference_dtype,
    }
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


def command_build_context(args):
    """convert -> build-context: sinh QNN context binary (.bin) tu .dlc,
    dung 1 lan tren host/CI (xem scripts/build_context_binary.sh - logic
    o day dua tren dung script do, boc lai thanh subcommand co input/
    output chuan hoa va chon duoc backend cpu/gpu/htp)."""
    qairt_root = Path(args.qairt_root)
    target = args.target
    paths = qairt_paths(qairt_root, target)
    dlc = Path(args.dlc).resolve()
    output = Path(args.output).resolve()

    ensure_exists(dlc, 'input DLC')
    output.parent.mkdir(parents=True, exist_ok=True)

    backend_lib = backend_lib_path(qairt_root, target, args.backend)
    ensure_file(paths['qnn_context_binary_generator'], 'qnn-context-binary-generator')
    ensure_file(backend_lib, f"{args.backend} backend library ({backend_lib.name})")
    ensure_file(paths['qnn_model_dlc'], 'libQnnModelDlc.so')

    binary_stem = output.stem
    env = make_env(qairt_root, target)

    cmd = [
        paths['qnn_context_binary_generator'],
        '--backend', backend_lib,
        '--model', paths['qnn_model_dlc'],
        '--dlc_path', dlc,
        '--binary_file', binary_stem,
        '--output_dir', output.parent,
    ]

    log_path = output.with_suffix('.build_context.log')
    run(cmd, env=env, log_path=log_path)

    generated = output.parent / f'{binary_stem}.bin'
    if generated != output:
        ensure_file(generated, 'generated context binary')
        shutil.move(str(generated), str(output))
    ensure_file(output, 'context binary')

    result = {
        'status': 'success',
        'context_bin': str(output),
        'backend': args.backend,
        'target': target,
        'log_path': str(log_path),
    }
    print(json.dumps(result, indent=2))
    return result


def command_run_api(args):
    """build-context -> run-api: build libai_runtime.a + traffic_app voi
    BACKEND=qnn_api (goi QNN C API that qua backend_qnn_api.c, KHONG qua
    qnn-net-run), roi chay traffic_app tren context binary + input raw,
    tra ve label/scores parse tu stdout cua application."""
    qairt_root = Path(args.qairt_root)
    target = args.target
    if target != DEFAULT_TARGET:
        fail('run-api executes traffic_app locally and currently supports only '
             f'{DEFAULT_TARGET}; use prepare to build artifacts for another device target')
    repo_root = Path(__file__).resolve().parent.parent
    context_bin = Path(args.context).resolve()
    input_raw = Path(args.input_raw).resolve()

    ensure_exists(context_bin, 'context binary (.bin)')
    ensure_exists(input_raw, 'input raw')

    paths = qairt_paths(qairt_root, target)
    backend_lib = backend_lib_path(qairt_root, target, args.backend)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    build_dir = work_dir / 'build'
    make_cmd = [
        'make', 'BACKEND=qnn_api', f'DL_QAIRT_ROOT={qairt_root}',
        f'BUILD_DIR={build_dir}',
    ]
    if args.cc:
        make_cmd.append(f'CC={args.cc}')
    if args.ar:
        make_cmd.append(f'AR={args.ar}')
    make_cmd.append('all')

    make_log = work_dir / 'make.log'
    run(make_cmd, cwd=repo_root, log_path=make_log)

    app_path = build_dir / 'traffic_app'
    ensure_file(app_path, 'traffic_app (qnn_api build)')

    env = os.environ.copy()
    env['DL_QAIRT_ROOT'] = str(qairt_root)
    env['DL_MODEL_PATH'] = str(context_bin)
    env['DL_INPUT_RAW'] = str(input_raw)
    if args.graph_name:
        env['DL_GRAPH_NAME'] = args.graph_name
    env['DL_QNN_BACKEND_LIB'] = str(backend_lib)
    env['DL_QNN_SYSTEM_LIB'] = str(paths['qnn_system'])
    old_ld = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f"{paths['lib']}:{old_ld}" if old_ld else str(paths['lib'])

    proc = run_capture([str(app_path)], env=env, cwd=repo_root)
    app_log = work_dir / 'traffic_app.log'
    app_log.write_text(f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n')

    if proc.returncode != 0:
        fail(f'traffic_app exited with code {proc.returncode}, see log: {app_log}')

    label, scores, latency_ms = parse_traffic_app_output(proc.stdout)
    if label is None or not scores:
        fail(f'failed to parse traffic_app output, see log: {app_log}')

    result = {
        'status': 'success',
        'label': label,
        'scores': scores,
        'latency_ms': latency_ms,
        'backend': args.backend,
        'target': target,
        'log_path': str(app_log),
    }
    print(json.dumps(result_with_limited_scores(
        result, getattr(args, 'max_print_scores', 100)), indent=2))
    return result


def command_inspect_context(args):
    """Build the direct-QNN metadata inspector and emit machine-readable JSON."""
    qairt_root = Path(args.qairt_root).expanduser().resolve()
    if args.target != DEFAULT_TARGET:
        fail(f'inspect-context executes locally and supports only {DEFAULT_TARGET}')
    context_bin = Path(args.context).expanduser().resolve()
    ensure_file(context_bin, 'context binary')
    repo_root = Path(__file__).resolve().parent.parent
    paths = qairt_paths(qairt_root, args.target)
    backend_lib = backend_lib_path(qairt_root, args.target, args.backend)
    ensure_file(backend_lib, f'{args.backend} backend library')
    ensure_file(paths['qnn_system'], 'QNN System library')

    work_dir = Path(args.work_dir).expanduser().resolve()
    build_dir = work_dir / 'build'
    run(['make', 'BACKEND=qnn_api', f'DL_QAIRT_ROOT={qairt_root}',
         f'BUILD_DIR={build_dir}', 'inspect'], cwd=repo_root,
        log_path=work_dir / 'make.log')
    inspector = build_dir / 'inspect_model'
    ensure_file(inspector, 'inspect_model executable')

    env = make_env(qairt_root, args.target)
    env['DL_MODEL_PATH'] = str(context_bin)
    env['DL_QNN_BACKEND_LIB'] = str(backend_lib)
    env['DL_QNN_SYSTEM_LIB'] = str(paths['qnn_system'])
    if args.graph_name:
        env['DL_GRAPH_NAME'] = args.graph_name
    proc = run_capture([str(inspector)], env=env, cwd=repo_root)
    log_path = work_dir / 'inspect.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n')
    if proc.returncode != 0:
        fail(f'inspect_model exited with code {proc.returncode}, see log: {log_path}')
    try:
        metadata = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        fail(f'inspect_model returned invalid JSON ({exc}), see log: {log_path}')
    result = {'status': 'success', 'context': str(context_bin), 'backend': args.backend,
              'target': args.target, 'metadata': metadata, 'log_path': str(log_path)}
    print(json.dumps(result, indent=2))
    return result


def command_run_context(args):
    """Execute a Context Binary through the generic multi-tensor C API."""
    qairt_root = Path(args.qairt_root).expanduser().resolve()
    if args.target != DEFAULT_TARGET:
        fail(f'run-context executes locally and supports only {DEFAULT_TARGET}')
    context_bin = Path(args.context).expanduser().resolve()
    ensure_file(context_bin, 'context binary')
    input_paths = [Path(value).expanduser().resolve() for value in args.input_raw]
    for index, path in enumerate(input_paths):
        ensure_file(path, f'input tensor {index}')

    repo_root = Path(__file__).resolve().parent.parent
    paths = qairt_paths(qairt_root, args.target)
    backend_lib = backend_lib_path(qairt_root, args.target, args.backend)
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = work_dir / 'build'
    run(['make', 'BACKEND=qnn_api', f'DL_QAIRT_ROOT={qairt_root}',
         f'BUILD_DIR={build_dir}', 'run-model'], cwd=repo_root,
        log_path=work_dir / 'make.log')
    executable = build_dir / 'run_model'
    ensure_file(executable, 'run_model executable')

    env = make_env(qairt_root, args.target)
    env['DL_MODEL_PATH'] = str(context_bin)
    env['DL_QNN_BACKEND_LIB'] = str(backend_lib)
    env['DL_QNN_SYSTEM_LIB'] = str(paths['qnn_system'])
    env['DL_OUTPUT_DIR'] = str(output_dir)
    if args.graph_name:
        env['DL_GRAPH_NAME'] = args.graph_name
    for index, path in enumerate(input_paths):
        env[f'DL_INPUT_{index}_RAW'] = str(path)

    proc = run_capture([str(executable)], env=env, cwd=repo_root)
    log_path = work_dir / 'run_context.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n')
    if proc.returncode != 0:
        fail(f'run_model exited with code {proc.returncode}, see log: {log_path}')
    try:
        execution = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        fail(f'run_model returned invalid JSON ({exc}), see log: {log_path}')
    for item in execution.get('outputs', []):
        item['path'] = str(output_dir / item['path'])
    result = {'status': 'success', 'context': str(context_bin), 'backend': args.backend,
              'target': args.target, **execution, 'log_path': str(log_path)}
    print(json.dumps(result, indent=2))
    return result


def command_prepare(args):
    """Minimal model -> DLC -> context-binary workflow, without inference."""
    qairt_root = Path(args.qairt_root).expanduser().resolve()
    model = model_from_args(args)
    ensure_file(model, 'input model')

    work_dir_value = getattr(args, 'work_dir', None)
    work_dir = (Path(work_dir_value).expanduser().resolve() if work_dir_value
                else (DEFAULT_WORK_ROOT / model.stem).resolve())
    work_dir.mkdir(parents=True, exist_ok=True)

    dlc_path = work_dir / f'{model.stem}.dlc'
    convert_result = command_convert(argparse.Namespace(
        qairt_root=str(qairt_root), model=str(model), model_positional=None,
        output=str(dlc_path), converter=args.converter,
        source_model_input_shape=args.source_model_input_shape,
        pytorch_input_dim=getattr(args, 'pytorch_input_dim', None),
        out_tensor_node=args.out_tensor_node,
        extra_args=normalized_extra_args(args.extra_args),
    ))

    context_path = work_dir / f'{model.stem}.{args.backend}.bin'
    context_result = command_build_context(argparse.Namespace(
        qairt_root=str(qairt_root), dlc=convert_result['dlc_path'],
        output=str(context_path), backend=args.backend, target=args.target,
    ))

    result = {
        'status': 'success',
        'source_model': str(model),
        'dlc_path': convert_result['dlc_path'],
        'context_bin': context_result['context_bin'],
        'backend': args.backend,
        'target': args.target,
        'work_dir': str(work_dir),
    }
    print(json.dumps(result, indent=2))
    return result


def command_deploy(args):
    """Chain toan bo: convert -> build-context -> run-api -> so sanh
    reference neu co. Day la lenh 'mot cua' theo dung muc tieu: dua model
    (.onnx/.tflite/.dlc) vao, ra ket qua da chay qua API runtime that,
    khong qua qnn-net-run."""
    qairt_root = Path(args.qairt_root)
    target = args.target
    model = Path(args.model).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(model, 'input model')

    dlc_path = work_dir / 'model.dlc'
    convert_args = argparse.Namespace(
        qairt_root=str(qairt_root),
        model=str(model),
        output=str(dlc_path),
        converter=args.converter,
        source_model_input_shape=args.source_model_input_shape,
        pytorch_input_dim=getattr(args, 'pytorch_input_dim', None),
        out_tensor_node=args.out_tensor_node,
        extra_args=args.extra_args or [],
    )
    command_convert(convert_args)

    context_bin_path = work_dir / 'model.bin'
    build_context_args = argparse.Namespace(
        qairt_root=str(qairt_root),
        dlc=str(dlc_path),
        output=str(context_bin_path),
        backend=args.backend,
        target=target,
    )
    command_build_context(build_context_args)

    run_api_args = argparse.Namespace(
        qairt_root=str(qairt_root),
        context=str(context_bin_path),
        input_raw=args.input_raw,
        backend=args.backend,
        target=target,
        graph_name=args.graph_name,
        work_dir=str(work_dir / 'run_api'),
        cc=args.cc,
        ar=args.ar,
    )
    run_result = command_run_api(run_api_args)

    if args.reference:
        ref_path = Path(args.reference).resolve()
        ensure_file(ref_path, 'reference output')

        if ref_path.suffix.lower() == '.npy':
            ref_values = read_npy_float32(ref_path)
        elif ref_path.suffix.lower() == '.raw':
            ref_values = read_float32(ref_path)
        else:
            fail('reference must be .npy or .raw')

        qnn = run_result['scores']
        n = min(len(qnn), len(ref_values))
        diffs = [abs(qnn[i] - ref_values[i]) for i in range(n)]
        max_abs_diff = max(diffs) if diffs else float('inf')
        mean_abs_diff = sum(diffs) / len(diffs) if diffs else float('inf')
        qnn_argmax = max(range(len(qnn)), key=lambda i: qnn[i]) if qnn else -1
        ref_argmax = max(range(len(ref_values)), key=lambda i: ref_values[i]) if ref_values else -1
        passed = len(qnn) == len(ref_values) and max_abs_diff <= args.tolerance and qnn_argmax == ref_argmax

        result = {
            'status': 'success' if passed else 'failed',
            'validation_passed': passed,
            'dlc_path': str(dlc_path),
            'context_bin': str(context_bin_path),
            'qnn_shape': [len(qnn)],
            'reference_shape': [len(ref_values)],
            'max_abs_diff': max_abs_diff,
            'mean_abs_diff': mean_abs_diff,
            'qnn_argmax': qnn_argmax,
            'reference_argmax': ref_argmax,
            'tolerance': args.tolerance,
            'latency_ms': run_result.get('latency_ms'),
        }
        print(json.dumps(result, indent=2))
        if not passed:
            raise SystemExit(2)
    else:
        result = {
            'status': 'success',
            'dlc_path': str(dlc_path),
            'context_bin': str(context_bin_path),
            'label': run_result['label'],
            'scores': run_result['scores'],
            'latency_ms': run_result.get('latency_ms'),
        }
        print(json.dumps(result_with_limited_scores(
            result, getattr(args, 'max_print_scores', 100)), indent=2))


def _write_embedded_model_c(model: Path, destination: Path):
    """Emit a C object instead of relying on objcopy being installed for ARM64."""
    data = model.read_bytes()
    with destination.open('w', encoding='ascii') as stream:
        stream.write('#include <stddef.h>\n')
        stream.write('const unsigned char kEmbeddedModel[] = {\n')
        for offset in range(0, len(data), 12):
            stream.write('  ' + ', '.join(f'0x{byte:02x}' for byte in data[offset:offset + 12]) + ',\n')
        stream.write('};\n')
        stream.write(f'const size_t kEmbeddedModelSize = {len(data)}u;\n')


def command_standalone_tflite(args):
    """Build one static ARM64 ELF: embedded .tflite + TFLite CPU runtime.

    This intentionally does not package QNN/Delegate libraries.  It is the
    deployment route for a generic ARM Linux target (including the iGate),
    where QNN drivers are absent.  Inputs remain files/sensors at runtime;
    only the model and inference runtime are compiled into the executable.
    """
    model = Path(args.model).expanduser().resolve()
    tensorflow_root = Path(args.tensorflow_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ensure_file(model, 'TFLite model')
    if model.suffix.lower() != '.tflite':
        fail('standalone-tflite accepts a .tflite file; convert ONNX/PT to TFLite before this CPU-only route')
    if not (tensorflow_root / 'WORKSPACE').is_file():
        fail(f'TensorFlow workspace not found: {tensorflow_root}')
    tflite_project = Path(__file__).resolve().parents[2] / 'tflite_qnn_prototype'
    runner_template = tflite_project / 'tools' / 'standalone_api_runner.c'
    app_source = Path(args.app_source).expanduser().resolve() if args.app_source else runner_template
    ensure_file(runner_template, 'standalone runner template')
    if output.exists() and not args.force:
        fail(f'output already exists: {output} (use --force to replace it)')

    digest = hashlib.sha256(model.read_bytes()).hexdigest()[:16]
    package_name = f'model_deploy_standalone_{digest}'
    package_dir = tensorflow_root / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    try:
        # Compile the existing AI Runtime layer into the payload.  The
        # application template includes ai_runtime.h only; TensorFlow Lite
        # remains an implementation detail of backend_tflite_delegate.c.
        for source, destination in (
            (app_source, 'standalone_runner.c'),
            (tflite_project / 'src' / 'ai_runtime.c', 'ai_runtime.c'),
            (tflite_project / 'src' / 'backend_tflite_delegate.c', 'backend_tflite_delegate.c'),
            (tflite_project / 'src' / 'backend.h', 'backend.h'),
            (tflite_project / 'include' / 'ai_runtime.h', 'ai_runtime.h'),
        ):
            ensure_file(source, 'standalone runtime source')
            shutil.copy2(source, package_dir / destination)
        _write_embedded_model_c(model, package_dir / 'model_data.c')
        (package_dir / 'BUILD').write_text(
            'cc_binary(\n'
            '    name = "runner",\n'
            '    srcs = ["standalone_runner.c", "ai_runtime.c", "backend_tflite_delegate.c", "model_data.c", "ai_runtime.h", "backend.h"],\n'
            '    deps = ["//tensorflow/lite/c:c_api"],\n'
            '    copts = ["-I.", "-DDL_TFLITE_STATIC_CPU"],\n'
            '    linkstatic = True,\n'
            '    # Strip symbols in the target linker: the payload is copied to\n'
            '    # a constrained device, not used as a host-side debug binary.\n'
            '    linkopts = ["-static", "-s", "-lm"],\n'
            ')\n', encoding='utf-8')
        bazel = args.bazel or str(Path.home() / 'bin' / 'bazelisk')
        command = [bazel]
        # Some WSL/CI environments cannot keep Bazel's gRPC server alive.
        # Batch mode is slower but uses no persistent server and is therefore
        # a reliable option for a reproducible deployment build.
        if getattr(args, 'bazel_batch', False):
            command.append('--batch')
        command.extend(['build', '-c', 'opt'])
        if args.target_config:
            command.append(f'--config={args.target_config}')
        command.append(f'//{package_name}:runner')
        run(command, cwd=tensorflow_root, log_path=output.parent / f'{output.name}.build.log')
        built = tensorflow_root / 'bazel-bin' / package_name / 'runner'
        ensure_file(built, 'static standalone executable')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        shutil.copy2(built, output)
        output.chmod(output.stat().st_mode | 0o111)
        readelf = shutil.which('readelf')
        if readelf:
            check = subprocess.run([readelf, '-d', str(output)], capture_output=True, text=True)
            if check.returncode != 0 or 'NEEDED' in check.stdout:
                fail('result is not fully static; see build log and target toolchain configuration')
        print(json.dumps({
            'status': 'success',
            'mode': 'tflite_cpu_static',
            'model_embedded': str(model),
            'executable': str(output),
            'usage': (f'{output.name} --output-dir /tmp/results INPUT_0.raw [INPUT_1.raw ...]'
                      if not args.app_source else f'{output.name}  # behavior defined by --app-source'),
            'note': 'CPU-only: no QNN/HTP acceleration and no external .so/.tflite is required on target.',
        }, indent=2))
    finally:
        if not args.keep_build_dir:
            shutil.rmtree(package_dir, ignore_errors=True)


def _quantize_tflite_weights(model: Path, tensorflow_root: Path, mode: str,
                             bazel: str, log_path: Path):
    """Create an INT8/FP16 weight-quantized TFLite model.

    This post-training transformation operates on an existing .tflite file and
    intentionally preserves its float input/output interface.  It is therefore
    different from calibrated full-integer quantization, which requires the
    original converter input and a representative dataset.
    """
    digest = hashlib.sha256(model.read_bytes()).hexdigest()[:16]
    package_name = f'model_deploy_quantize_{digest}_{mode}'
    package_dir = tensorflow_root / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()

    quantized_model = package_dir / f'model_{mode}.tflite'
    buffer_type = {
        'int8': 'QUANTIZED_INT8',
        'float16': 'QUANTIZED_FLOAT16',
    }[mode]
    source = f'''#include <fstream>
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/tools/optimize/quantize_weights.h"

int main(int argc, char **argv) {{
  if (argc != 3) return 2;
  auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  if (!model || model->GetModel() == nullptr) return 3;
  flatbuffers::FlatBufferBuilder builder;
  if (tflite::optimize::QuantizeWeights(
          &builder, model->GetModel(),
          tflite::optimize::BufferType::{buffer_type}) != kTfLiteOk) {{
    return 4;
  }}
  std::ofstream out(argv[2], std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
            builder.GetSize());
  return out.good() ? 0 : 5;
}}
'''
    (package_dir / 'quantize_tflite.cc').write_text(source, encoding='utf-8')
    (package_dir / 'BUILD').write_text(
        '# Use TensorFlow Lite\'s portable weight quantizer directly. Depending\n'
        '# on the generic quantize_weights target would also build the MLIR\n'
        '# quantizer and a large host-only dependency graph.\n'
        'cc_library(\n'
        '    name = "portable_weight_quantizer",\n'
        '    srcs = ["//tensorflow/lite/tools/optimize:quantize_weights_portable.cc"],\n'
        '    hdrs = ["//tensorflow/lite/tools/optimize:quantize_weights.h"],\n'
        '    deps = [\n'
        '        "//tensorflow/lite/tools/optimize:quantization_utils",\n'
        '        "//tensorflow/lite/tools/optimize:model_utils",\n'
        '        "@com_google_absl//absl/memory",\n'
        '        "@com_google_absl//absl/strings",\n'
        '        "@com_google_absl//absl/container:flat_hash_map",\n'
        '        "@com_google_absl//absl/container:flat_hash_set",\n'
        '        "@flatbuffers",\n'
        '        "//tensorflow/lite:framework",\n'
        '        "//tensorflow/lite/core:framework",\n'
        '        "//tensorflow/lite/kernels/internal:tensor_utils",\n'
        '        "//tensorflow/lite/schema:schema_fbs",\n'
        '        "//tensorflow/lite/schema:schema_utils",\n'
        '        "//tensorflow/core:tflite_portable_logging",\n'
        '    ],\n'
        ')\n\n'
        'cc_binary(\n'
        '    name = "quantize_tflite",\n'
        '    srcs = ["quantize_tflite.cc"],\n'
        '    deps = [\n'
        '        "//tensorflow/lite/core:model_builder",\n'
        '        ":portable_weight_quantizer",\n'
        '        "@flatbuffers",\n'
        '    ],\n'
        ')\n', encoding='utf-8')

    target = f'//{package_name}:quantize_tflite'
    run([bazel, 'build', '-c', 'opt', target], cwd=tensorflow_root,
        log_path=log_path)
    executable = tensorflow_root / 'bazel-bin' / package_name / 'quantize_tflite'
    ensure_file(executable, 'TFLite weight quantizer')
    run([str(executable), str(model), str(quantized_model)],
        cwd=tensorflow_root,
        log_path=log_path.with_name(f'{log_path.stem}.run{log_path.suffix}'))
    ensure_file(quantized_model, f'{mode} weight-quantized TFLite model')
    return quantized_model, package_dir


def _generated_tflite_path(output: Path) -> Path:
    """Choose the inspectable TFLite artifact beside a packaged library."""
    return output.with_suffix('.source.tflite')


def _convert_onnx_to_tflite(args, model: Path, destination: Path) -> Path:
    """Best-effort ONNX -> TFLite conversion through the isolated ONNX venv.

    ONNX is intentionally not treated as a guarantee: unsupported operations
    or dynamic graph constructs need a re-export from the model owner.  The
    complete converter log is retained beside the generated TFLite model.
    """
    converter = Path(args.onnx2tf).expanduser().resolve()
    ensure_file(converter, 'onnx2tf executable')
    work_dir = destination.with_suffix('.onnx2tf-work')
    if work_dir.exists():
        shutil.rmtree(work_dir)
    command = [str(converter), '-i', str(model), '-o', str(work_dir), '-nuo', '--non_verbose']
    if args.onnx_input_shape:
        command.extend(['-ois', *args.onnx_input_shape])
    run(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
        log_path=destination.with_suffix('.onnx2tf.log'))

    candidates = sorted(work_dir.rglob('*.tflite'))
    if not candidates:
        fail('onnx2tf finished without producing a .tflite file; see log: '
             f'{destination.with_suffix(".onnx2tf.log")}')
    float32 = [path for path in candidates if 'float32' in path.name.lower()]
    selected = float32[0] if len(float32) == 1 else candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, destination)
    shutil.rmtree(work_dir, ignore_errors=True)
    return destination


def _convert_pytorch_to_tflite(args, model: Path, destination: Path) -> Path:
    """Invoke the separate PyTorch/LiteRT environment without importing it here."""
    # Do not resolve this path: a venv's ``bin/python`` commonly is a symlink
    # to the system interpreter.  Resolving it would discard pyvenv.cfg and
    # execute outside the isolated model-torch environment.
    python = Path(args.torch_python).expanduser()
    ensure_file(python, 'model-torch Python executable')
    exporter = Path(__file__).resolve().with_name('pytorch_to_tflite.py')
    ensure_file(exporter, 'PyTorch TFLite exporter')
    command = [str(python), str(exporter), '--model', str(model), '--output', str(destination)]
    if args.input_shape:
        command.extend(['--input-shape', args.input_shape])
    if args.metadata:
        command.extend(['--metadata', args.metadata])
    if args.reference_input:
        command.extend(['--reference-input', args.reference_input])
    if args.model_loader:
        command.extend(['--model-loader', args.model_loader])
    if args.trust_pytorch_pickle:
        command.append('--trust-pytorch-pickle')
    run(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
        log_path=destination.with_suffix('.pytorch-export.log'))
    ensure_file(destination, 'exported TFLite model')
    return destination


def command_package_deep_learning(args):
    """Turn TFLite, ONNX, or supported PyTorch input into an ARM64 model .so."""
    source = Path(args.model).expanduser().resolve()
    ensure_file(source, 'source model')
    suffix = source.suffix.lower()
    output = Path(args.output).expanduser().resolve()
    if suffix not in ('.tflite', '.onnx', '.pt', '.pth'):
        fail('package-deep-learning accepts .tflite, .onnx, .pt, or .pth')
    if output.suffix.lower() != '.so':
        fail('package-deep-learning always creates a shared library; --output must end in .so')

    if suffix == '.tflite':
        tflite = source
        conversion = 'none'
    else:
        tflite = _generated_tflite_path(output)
        if tflite.exists() and not args.force:
            fail(f'generated TFLite already exists: {tflite} (use --force to replace it)')
        if suffix == '.onnx':
            tflite = _convert_onnx_to_tflite(args, source, tflite)
            conversion = 'onnx2tf'
        else:
            tflite = _convert_pytorch_to_tflite(args, source, tflite)
            conversion = 'pytorch-litert'

    original_model = args.model
    args.model = str(tflite)
    try:
        command_static_library(args)
    finally:
        args.model = original_model
    print(json.dumps({
        'status': 'success',
        'source_model': str(source),
        'source_format': suffix.removeprefix('.'),
        'conversion': conversion,
        'tflite_model': str(tflite),
        'library': str(output),
    }, indent=2))


def command_static_library(args):
    """Build a reusable static archive containing one embedded TFLite model.

    The archive exports ai_model_init/predict/deinit.  TensorFlow Lite's
    transitive static archive is copied alongside it so an application can
    link both archives without knowing the generated Bazel package name.
    """
    model = Path(args.model).expanduser().resolve()
    tensorflow_root = Path(args.tensorflow_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    ensure_file(model, 'TFLite model')
    if model.suffix.lower() != '.tflite':
        fail('static-library accepts a .tflite file')
    if not (tensorflow_root / 'WORKSPACE').is_file():
        fail(f'TensorFlow workspace not found: {tensorflow_root}')
    if output.exists() and not args.force:
        fail(f'output already exists: {output} (use --force to replace it)')

    tflite_project = Path(__file__).resolve().parents[2] / 'tflite_qnn_prototype'
    bazel = args.bazel or str(Path.home() / 'bin' / 'bazelisk')
    quantize_mode = getattr(args, 'quantize', 'none')
    effective_model = model
    quantize_dir = None
    if quantize_mode != 'none':
        effective_model, quantize_dir = _quantize_tflite_weights(
            model, tensorflow_root, quantize_mode, bazel,
            output.parent / f'{output.name}.quantize.log')

    digest = hashlib.sha256(effective_model.read_bytes()).hexdigest()[:16]
    package_name = f'model_deploy_library_{digest}'
    package_dir = tensorflow_root / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    try:
        for source, destination in (
            (tflite_project / 'src' / 'ai_model.c', 'ai_model.c'),
            (tflite_project / 'src' / 'ai_runtime.c', 'ai_runtime.c'),
            (tflite_project / 'src' / 'backend_tflite_delegate.c', 'backend_tflite_delegate.c'),
            (tflite_project / 'src' / 'backend.h', 'backend.h'),
            (tflite_project / 'include' / 'ai_model.h', 'ai_model.h'),
            (tflite_project / 'include' / 'ai_runtime.h', 'ai_runtime.h'),
        ):
            ensure_file(source, 'static library source')
            shutil.copy2(source, package_dir / destination)
        # Keep the flatbuffer as a Bazel data input for TensorFlow Lite's
        # model-selective operator registration, and also embed the same bytes
        # in model_data.c for deployment.
        shutil.copy2(effective_model, package_dir / 'embedded_model.tflite')
        _write_embedded_model_c(effective_model, package_dir / 'model_data.c')
        if args.shared:
            build_rule = (
                'load("//tensorflow/lite:build_def.bzl", "tflite_custom_c_library")\n\n'
                '# Generate a C API runtime containing only operators used by\n'
                '# embedded_model.tflite.\n'
                'tflite_custom_c_library(\n'
                '    name = "model_c_api",\n'
                '    models = ["embedded_model.tflite"],\n'
                ')\n\n'
                'cc_binary(\n'
                '    name = "ai_model",\n'
                '    srcs = ["ai_model.c", "ai_runtime.c", "backend_tflite_delegate.c", "model_data.c", "ai_model.h", "ai_runtime.h", "backend.h"],\n'
                '    deps = [":model_c_api"],\n'
                '    copts = ["-I.", "-DDL_TFLITE_STATIC_CPU", "-Os", "-ffunction-sections", "-fdata-sections"],\n'
                '    # TFLite is implemented in C++; embed its C++/GCC runtime so\n'
                '    # the deployed .so works on minimal ARM64 firmware images.\n'
                '    # The embedded ARM toolchain adds -lstdc++ after ordinary\n'
                '    # linkopts.  Select its static archive explicitly as well, so\n'
                '    # the final .so does not require libstdc++.so.6 on the board.\n'
                '    linkopts = [\n'
                '        "-Wl,--as-needed",\n'
                '        "-static-libstdc++", "-static-libgcc",\n'
                '        "-Wl,--push-state,-Bstatic", "-lstdc++",\n'
                '        "-Wl,--pop-state",\n'
                '        "-Wl,--gc-sections",\n'
                '        "-Wl,--version-script,$(location :exports.lds)",\n'
                '    ],\n'
                '    additional_linker_inputs = ["exports.lds"],\n'
                '    linkshared = True,\n'
                ')\n')
        else:
            build_rule = (
                'cc_library(\n'
                '    name = "ai_model",\n'
                '    srcs = ["ai_model.c", "ai_runtime.c", "backend_tflite_delegate.c", "model_data.c"],\n'
                '    hdrs = ["ai_model.h", "ai_runtime.h", "backend.h"],\n'
                '    deps = ["//tensorflow/lite/c:c_api"],\n'
                '    copts = ["-I.", "-DDL_TFLITE_STATIC_CPU"],\n'
                '    linkstatic = True,\n'
                ')\n')
        (package_dir / 'BUILD').write_text(build_rule, encoding='utf-8')
        if args.shared:
            # The application-facing wrapper is the only public ABI. Hiding
            # TFLite/C++ implementation symbols cuts the dynamic symbol table
            # and prevents applications from depending on backend internals.
            (package_dir / 'exports.lds').write_text(
                '{\n'
                '  global:\n'
                '    ai_model_init;\n'
                '    ai_model_get_io_count;\n'
                '    ai_model_predict;\n'
                '    ai_model_get_tensor_count;\n'
                '    ai_model_get_input_info;\n'
                '    ai_model_get_output_info;\n'
                '    ai_model_predict_tensors;\n'
                '    ai_model_deinit;\n'
                '  local: *;\n'
                '};\n', encoding='utf-8')
        command = [bazel]
        if getattr(args, 'bazel_batch', False):
            command.append('--batch')
        command.extend(['build', '-c', 'opt'])
        if args.aarch64_staging_dir:
            staging_dir = Path(args.aarch64_staging_dir).expanduser().resolve()
            if not staging_dir.is_dir():
                fail(f'ARM64 staging directory not found: {staging_dir}')
            # OpenWrt compiler wrappers use STAGING_DIR to find their target
            # headers and libraries. Bazel sanitizes the action environment,
            # so exporting it only in the caller shell is insufficient.
            command.append(f'--action_env=STAGING_DIR={staging_dir}')
        if args.aarch64_toolchain:
            toolchain = Path(args.aarch64_toolchain).expanduser().resolve()
            # Preserve an OpenWrt compiler symlink name.  Its wrapper selects
            # gcc/g++ from argv[0], so resolving it to the shared wrapper
            # script changes its behaviour.
            compiler = (Path(os.path.abspath(os.fspath(
                        Path(args.aarch64_compiler).expanduser())))
                        if args.aarch64_compiler else
                        toolchain / 'bin' / 'aarch64-none-linux-gnu-gcc')
            ensure_file(compiler, 'ARM64 cross compiler')
            if not ((toolchain / 'BUILD').is_file() or
                    (toolchain / 'BUILD.bazel').is_file()):
                fail(f'Bazel repository BUILD file not found in toolchain: {toolchain}')
            command.append(f'--override_repository=aarch64_linux_toolchain={toolchain}')
            if args.aarch64_toolchain_config:
                toolchain_config = Path(args.aarch64_toolchain_config).expanduser().resolve()
                ensure_file(toolchain_config / 'cc_config.bzl',
                            'Bazel ARM64 toolchain configuration')
                if not ((toolchain_config / 'BUILD').is_file() or
                        (toolchain_config / 'BUILD.bazel').is_file()):
                    fail('Bazel toolchain configuration BUILD file not found: '
                         f'{toolchain_config}')
                command.append(
                    '--override_repository=local_config_embedded_arm='
                    f'{toolchain_config}')
        if not args.shared:
            command.extend(['--define', 'framework_shared_object=false'])
        if args.target_libc == 'musl':
            # TensorFlow v2.15's vendored FlatBuffers infers the availability
            # of glibc's strto*_l functions from _XOPEN_SOURCE.  musl does
            # not provide those functions, so force FlatBuffers to use its
            # portable strto* fallback for a musl target.
            command.append('--copt=-DFLATBUFFERS_LOCALE_INDEPENDENT=0')
        if args.target_config:
            command.append(f'--config={args.target_config}')
        # TensorFlow's elinux config points --host_crosstool_top at the
        # generic Bazel tools suite. With newer vendor cross compilers,
        # exec-config dependencies such as FlatBuffers' `flatc` can then
        # incorrectly select the ARM64 target compiler. These overrides must
        # occur after --config so its expansion cannot replace them.
        if args.aarch64_toolchain and os.uname().machine in ('x86_64', 'amd64'):
            host_cc = shutil.which('gcc')
            host_cxx = shutil.which('g++')
            if not host_cc or not host_cxx:
                fail('native host gcc/g++ not found; required to build Bazel exec tools')
            command.extend([
                '--host_cpu=k8',
                '--host_crosstool_top=@local_config_cc//:toolchain',
                f'--repo_env=CC={host_cc}',
                f'--repo_env=CXX={host_cxx}',
            ])
        command.extend([
            '--copt=-Os',
            '--copt=-ffunction-sections',
            '--copt=-fdata-sections',
            '--linkopt=-Wl,--gc-sections',
            f'//{package_name}:ai_model',
        ])
        run(command, cwd=tensorflow_root, log_path=output.parent / f'{output.name}.build.log')
        built_name = 'libai_model.so' if args.shared else 'libai_model.a'
        built = tensorflow_root / 'bazel-bin' / package_name / built_name
        ensure_file(built, 'AI model library')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        shutil.copy2(built, output)
        # Bazel marks generated binaries read-only.  Preserve the useful
        # executable bits but make the copied artifact owner-writable so a
        # normal deployment step can run `strip --strip-unneeded OUTPUT`
        # without requiring a separate chmod workaround.
        output.chmod(output.stat().st_mode | 0o200)
        if not args.shared:
            tflite_archive = (tensorflow_root / 'bazel-bin' / 'tensorflow' / 'lite' /
                              'libtflite_with_xnnpack_optional.a')
            if tflite_archive.is_file():
                shutil.copy2(tflite_archive, output.parent / 'libtflite_with_xnnpack_optional.a')
        shutil.copy2(tflite_project / 'include' / 'ai_model.h', output.parent / 'ai_model.h')
        shutil.copy2(tflite_project / 'include' / 'ai_runtime.h', output.parent / 'ai_runtime.h')
        print(json.dumps({
            'status': 'success',
            'mode': 'tflite_cpu_shared_library' if args.shared else 'tflite_cpu_static_library',
            'library': str(output),
            'headers': [str(output.parent / 'ai_model.h'), str(output.parent / 'ai_runtime.h')],
            'runtime': 'model-selective' if args.shared else 'full-static-archive',
            'quantize': quantize_mode,
            'embedded_model_bytes': effective_model.stat().st_size,
            'note': ('Model bytes and TFLite runtime are embedded in the shared library; '
                     'link the application with -lai_model.' if args.shared else
                     'Static archive requires a separately linkable TFLite C API archive.'),
        }, indent=2))
    finally:
        if not args.keep_build_dir:
            shutil.rmtree(package_dir, ignore_errors=True)
            if quantize_dir is not None:
                shutil.rmtree(quantize_dir, ignore_errors=True)


def build_parser():
    parser = argparse.ArgumentParser(description='Model deploy helper for QAIRT/QNN prototype')
    parser.add_argument('--qairt-root', default=str(DEFAULT_QAIRT_ROOT))
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('doctor', help='Check SDK, converter, backend and target prerequisites')
    p.add_argument('model', nargs='?', metavar='MODEL')
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--htp-arch', help='Optional HTP architecture to validate Stub/Skel availability, e.g. v73')
    p.set_defaults(func=command_doctor)

    p = sub.add_parser('package-target',
                       help='Create a non-destructive Linux deployment bundle with QNN runtime libraries')
    p.add_argument('--artifact', required=True,
                   help='QNN Context Binary (.bin) to place in the direct-QNN runtime bundle')
    p.add_argument('--artifact-kind', choices=['context-binary', 'dlc', 'tflite'], default='context-binary')
    p.add_argument('--output', required=True, help='New bundle directory; must not already exist')
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', required=True,
                   help='QAIRT Linux ABI triplet, e.g. aarch64-oe-linux-gcc11.2')
    p.add_argument('--htp-arch', help='Required for HTP: v68, v69, v73, v75, v79, ...')
    p.add_argument('--app', help='Optional target-built runtime application to copy and execute via run.sh')
    p.set_defaults(func=command_package_target)

    p = sub.add_parser('standalone-tflite',
                       help='Build one static ARM64 ELF with an embedded .tflite model (CPU only)')
    p.add_argument('--model', required=True, help='Source .tflite model to compile into the executable')
    p.add_argument('--output', required=True, help='Output executable path, for example dist/model_runner')
    p.add_argument('--tensorflow-root', default=str(Path.home() / 'tensorflow'),
                   help='TensorFlow v2.15 source tree used to statically link the TFLite C API')
    p.add_argument('--bazel', default=None, help='Bazelisk/Bazel executable (default: ~/bin/bazelisk)')
    p.add_argument('--target-config', default='elinux_aarch64',
                   help='TensorFlow Bazel config for the target; use elinux_aarch64 for ARM64 Linux')
    p.add_argument('--app-source', default=None,
                   help='Optional C file containing main(); it may include only ai_runtime.h and call dl_* APIs')
    p.add_argument('--force', action='store_true', help='Replace an existing output executable')
    p.add_argument('--keep-build-dir', action='store_true',
                   help='Keep the temporary Bazel package under TensorFlow source for troubleshooting')
    p.set_defaults(func=command_standalone_tflite)

    p = sub.add_parser('static-library',
                       help='Build a reusable static archive with an embedded TFLite model')
    p.add_argument('--model', required=True, help='Source .tflite model')
    p.add_argument('--output', required=True, help='Output archive, for example dist/libai_model.a')
    p.add_argument('--tensorflow-root', default=str(Path.home() / 'tensorflow'))
    p.add_argument('--bazel', default=None)
    p.add_argument('--bazel-batch', action='store_true',
                   help='Run Bazel without its persistent server (slower, useful on WSL/CI)')
    p.add_argument('--target-config', default='elinux_aarch64')
    p.add_argument('--aarch64-toolchain', default=None,
                   help='Bazel repository containing an ARM64 GNU toolchain; '
                        'use a sysroot no newer than the target glibc')
    p.add_argument('--aarch64-compiler', default=None,
                   help='Cross GCC selected in the generated Bazel configuration; '
                        'required for vendor toolchains whose compiler is not '
                        'aarch64-none-linux-gnu-gcc')
    p.add_argument('--aarch64-toolchain-config', default=None,
                   help='Bazel local_config_embedded_arm repository matching '
                        'the selected ARM64 compiler version')
    p.add_argument('--aarch64-staging-dir', default=None,
                   help='OpenWrt/QSDK staging_dir passed into Bazel build actions')
    p.add_argument('--target-libc', choices=['glibc', 'musl'], default='glibc',
                   help='C library ABI of the ARM64 target (default: glibc)')
    p.add_argument('--shared', action='store_true',
                   help='Build libai_model.so containing the model and TFLite runtime')
    p.add_argument('--quantize', choices=['none', 'int8', 'float16'], default='none',
                   help='Optionally quantize model weights before embedding. '
                        'The int8/float16 modes preserve float model I/O and are '
                        'not calibrated full-integer quantization (default: none).')
    p.add_argument('--force', action='store_true')
    p.add_argument('--keep-build-dir', action='store_true')
    p.set_defaults(func=command_static_library)

    p = sub.add_parser('package-deep-learning',
                       help='Convert .tflite/.onnx/.pt/.pth when needed, then build an ARM64 model .so')
    p.add_argument('--model', required=True,
                   help='Source .tflite, .onnx, .pt, or .pth model')
    p.add_argument('--output', required=True,
                   help='Output shared library, for example dist/libai_model.so')
    p.add_argument('--tensorflow-root', default=str(Path.home() / 'tensorflow'))
    p.add_argument('--bazel', default=None)
    p.add_argument('--bazel-batch', action='store_true',
                   help='Run Bazel without its persistent server (slower, useful on WSL/CI)')
    p.add_argument('--target-config', default='elinux_aarch64')
    p.add_argument('--aarch64-toolchain', default=None)
    p.add_argument('--aarch64-compiler', default=None)
    p.add_argument('--aarch64-toolchain-config', default=None)
    p.add_argument('--aarch64-staging-dir', default=None)
    p.add_argument('--target-libc', choices=['glibc', 'musl'], default='glibc')
    p.add_argument('--quantize', choices=['none', 'int8', 'float16'], default='none')
    p.add_argument('--force', action='store_true')
    p.add_argument('--keep-build-dir', action='store_true')
    p.add_argument('--torch-python', default=str(Path.home() / 'venvs/model-torch/bin/python'),
                   help='Python executable in the isolated PyTorch/LiteRT environment')
    p.add_argument('--trust-pytorch-pickle', action='store_true',
                   help='Permit loading a trusted non-TorchScript .pt/.pth checkpoint')
    p.add_argument('--metadata',
                   help='Optional preprocessing metadata JSON for PyTorch models')
    p.add_argument('--input-shape',
                   help='Override model input shape, for example 1,90,3')
    p.add_argument('--reference-input',
                   help='Optional .npy input for PyTorch/TFLite parity validation')
    p.add_argument('--model-loader',
                   help='Trusted Python loader for a PyTorch state_dict checkpoint')
    p.add_argument('--onnx2tf', default=str(Path.home() / 'venvs/model-onnx/bin/onnx2tf'),
                   help='onnx2tf executable in the isolated ONNX environment')
    p.add_argument('--onnx-input-shape', action='append',
                   help='Static ONNX input override, for example input_seq:1,80,3; repeat for multiple inputs')
    # This command deliberately has one deployment artifact: a shared model
    # library.  Do not expose --shared here, because turning it off would make
    # the command's --output .so contract false.
    p.set_defaults(func=command_package_deep_learning, shared=True)

    p = sub.add_parser('convert', help='Convert MODEL to .dlc; output defaults next to MODEL')
    p.add_argument('model_positional', nargs='?', metavar='MODEL')
    p.add_argument('--model', help='Backward-compatible alternative to positional MODEL')
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--converter', choices=['auto', 'qairt', 'qnn', 'snpe'], default='auto')
    p.add_argument('--source-model-input-shape', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--pytorch-input-dim', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'),
                   help='Required for .pt/.pth TorchScript input, e.g. input 1,80,3')
    p.add_argument('--out-tensor-node', action='append')
    p.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[],
                   help='Arguments passed verbatim to the Qualcomm converter; place this option last')
    p.set_defaults(func=command_convert)

    p = sub.add_parser('prepare', help='One command: MODEL -> .dlc + QNN context .bin')
    p.add_argument('model_positional', nargs='?', metavar='MODEL')
    p.add_argument('--model', help='Backward-compatible alternative to positional MODEL')
    p.add_argument('-d', '--work-dir', default=None,
                   help='Output directory (default: /tmp/model_deploy_tool/<model-name>)')
    p.add_argument('--converter', choices=['auto', 'qairt', 'qnn', 'snpe'], default='auto')
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--source-model-input-shape', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--pytorch-input-dim', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--out-tensor-node', action='append')
    p.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[],
                   help='Arguments passed verbatim to the Qualcomm converter; place this option last')
    p.set_defaults(func=command_prepare)

    p = sub.add_parser('run-host', help='Run a DLC using QNN CPU backend on host')
    p.add_argument('--dlc', required=True)
    p.add_argument('--input-raw', required=True)
    p.add_argument('--input-name', default='input_seq')
    p.add_argument('--output-name', default='class_probs')
    p.add_argument('--output-dir', default=str(DEFAULT_WORK_ROOT / 'run_output'))
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'work'))
    p.add_argument('--log-level', default='error')
    p.set_defaults(func=command_run_host)

    p = sub.add_parser('validate', help='Run host inference and compare with reference output')
    p.add_argument('--dlc', required=True)
    p.add_argument('--input-raw', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--input-name', default='input_seq')
    p.add_argument('--output-name', default='class_probs')
    p.add_argument('--output-dir', default=str(DEFAULT_WORK_ROOT / 'validate_output'))
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'validate_work'))
    p.add_argument('--log-level', default='error')
    p.add_argument('--tolerance', type=float, default=1e-5)
    p.add_argument('--relative-tolerance', type=float, default=0.0)
    p.add_argument('--output-dtype', choices=['float32', 'uint8', 'int8'], default='float32')
    p.add_argument('--output-scale', type=float, default=1.0)
    p.add_argument('--output-zero-point', type=int, default=0)
    p.add_argument('--reference-dtype', choices=['float32', 'uint8', 'int8'], default='float32')
    p.add_argument('--reference-scale', type=float, default=1.0)
    p.add_argument('--reference-zero-point', type=int, default=0)
    p.add_argument('--require-argmax', action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=command_validate)

    p = sub.add_parser('build-context', help='Build a QNN context binary (.bin) from a .dlc')
    p.add_argument('--dlc', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET,
                    help='QAIRT target triplet the generator tool runs on (thuong van la host, '
                         'vi du x86_64-linux-clang, ke ca khi build context cho htp).')
    p.set_defaults(func=command_build_context)

    p = sub.add_parser('run-api', help='Build BACKEND=qnn_api va chay traffic_app tren 1 context binary')
    p.add_argument('--context', required=True, help='Duong dan context binary .bin (KHONG phai .dlc)')
    p.add_argument('--input-raw', required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET,
                    help=f'Local runtime SDK target; currently only {DEFAULT_TARGET} is executable')
    p.add_argument('--graph-name', default=None)
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'run_api_work'))
    p.add_argument('--cc', default=None, help='Override CC (vi du toolchain cross-compile)')
    p.add_argument('--ar', default=None, help='Override AR (vi du toolchain cross-compile)')
    p.add_argument('--max-print-scores', type=int, default=100,
                   help='Maximum scores emitted in JSON; validation still uses the full tensor')
    p.set_defaults(func=command_run_api)

    p = sub.add_parser('inspect-context', help='Read graph tensor metadata from a QNN Context Binary')
    p.add_argument('--context', required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--graph-name', default=None)
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'inspect_context'))
    p.set_defaults(func=command_inspect_context)

    p = sub.add_parser('run-context', help='Run a QNN Context Binary with one or more raw input tensors')
    p.add_argument('--context', required=True)
    p.add_argument('--input-raw', action='append', required=True,
                   help='Raw input in graph tensor order; repeat for every input')
    p.add_argument('--output-dir', default=str(DEFAULT_WORK_ROOT / 'run_context_output'))
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--graph-name', default=None)
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'run_context'))
    p.set_defaults(func=command_run_context)

    p = sub.add_parser('deploy', help='Chain convert -> build-context -> run-api (+ so sanh reference neu co)')
    p.add_argument('--model', required=True)
    p.add_argument('--input-raw', required=True)
    p.add_argument('--reference', default=None)
    p.add_argument('--converter', choices=['auto', 'qairt', 'qnn', 'snpe'], default='auto')
    p.add_argument('--source-model-input-shape', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--pytorch-input-dim', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--out-tensor-node', action='append')
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--graph-name', default=None)
    p.add_argument('--work-dir', default=str(DEFAULT_WORK_ROOT / 'deploy'))
    p.add_argument('--cc', default=None)
    p.add_argument('--ar', default=None)
    p.add_argument('--tolerance', type=float, default=1e-5)
    p.add_argument('--max-print-scores', type=int, default=100,
                   help='Maximum scores emitted in JSON; validation still uses the full tensor')
    p.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[],
                   help='Arguments passed verbatim to the Qualcomm converter; place this option last')
    p.set_defaults(func=command_deploy)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()




