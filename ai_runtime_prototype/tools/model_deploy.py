#!/usr/bin/env python3
import argparse
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
    check('qairt-converter', paths['qairt_converter'])
    check('qnn-context-binary-generator', paths['qnn_context_binary_generator'])
    check('qnn-net-run', paths['qnn_net_run'])
    check('libQnnSystem.so', paths['qnn_system'])
    check('libQnnModelDlc.so', paths['qnn_model_dlc'])
    check(f'libQnn{args.backend.capitalize()}.so',
          backend_lib_path(qairt_root, args.target, args.backend))

    if args.model:
        model = Path(args.model).expanduser().resolve()
        check('input model', model)
        extension = model.suffix.lower()
        if extension == '.onnx':
            check('qnn-onnx-converter', paths['qnn_onnx_converter'], required=False)
        elif extension == '.tflite':
            check('qnn-tflite-converter', paths['qnn_tflite_converter'], required=False)
        elif extension in ('.pt', '.pth'):
            check('qnn-pytorch-converter', paths['qnn_pytorch_converter'])
            is_torchscript = model.is_file() and is_torchscript_archive(model)
            checks.append({
                'name': 'TorchScript archive', 'path': str(model), 'required': True,
                'ok': is_torchscript,
                'hint': 'export a training checkpoint with torch.jit.trace/script before conversion',
            })
        elif extension != '.dlc':
            checks.append({
                'name': 'supported model extension', 'path': extension or '<none>', 'required': True,
                'ok': False, 'hint': 'use ONNX, TFLite, TorchScript .pt/.pth, or DLC',
            })

    missing = [item for item in checks if item['required'] and not item['ok']]
    warnings = []
    if args.target != DEFAULT_TARGET:
        warnings.append('This target can be prepared, but run-api executes only local x86_64 binaries. '
                        'Copy artifacts and run them on the matching device.')
    if args.backend == 'htp':
        warnings.append('HTP preflight cannot validate the target SoC, firmware, skeleton library, or graph partition. '
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
                     '--pytorch-input-dim input 1,240')
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
    qnn = read_float32(output_raw)
    ref_path = Path(args.reference).resolve()
    ensure_file(ref_path, 'reference output')

    if ref_path.suffix.lower() == '.npy':
        ref_values = read_npy_float32(ref_path)
    elif ref_path.suffix.lower() == '.raw':
        ref_values = read_float32(ref_path)
    else:
        fail('reference must be .npy or .raw')

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
        'qnn_shape': [len(qnn)],
        'reference_shape': [len(ref_values)],
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'qnn_argmax': qnn_argmax,
        'reference_argmax': ref_argmax,
        'tolerance': args.tolerance,
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


def build_parser():
    parser = argparse.ArgumentParser(description='Model deploy helper for QAIRT/QNN prototype')
    parser.add_argument('--qairt-root', default=str(DEFAULT_QAIRT_ROOT))
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('doctor', help='Check SDK, converter, backend and target prerequisites')
    p.add_argument('model', nargs='?', metavar='MODEL')
    p.add_argument('--backend', choices=['cpu', 'gpu', 'htp'], default='cpu')
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.set_defaults(func=command_doctor)

    p = sub.add_parser('convert', help='Convert MODEL to .dlc; output defaults next to MODEL')
    p.add_argument('model_positional', nargs='?', metavar='MODEL')
    p.add_argument('--model', help='Backward-compatible alternative to positional MODEL')
    p.add_argument('-o', '--output', default=None)
    p.add_argument('--converter', choices=['auto', 'qairt', 'qnn', 'snpe'], default='auto')
    p.add_argument('--source-model-input-shape', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'))
    p.add_argument('--pytorch-input-dim', nargs=2, action='append', metavar=('INPUT_NAME', 'INPUT_DIMS'),
                   help='Required for .pt/.pth TorchScript input, e.g. input 1,240')
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




