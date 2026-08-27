#!/usr/bin/env python3
"""
tools/regression_test.py

Chay lai pipeline 'deploy' (convert -> build-context -> run-api -> so sanh
reference) cho TAT CA cac model da biet la chay dung, va assert
validation_passed == true cho tung cai.

Ly do co script nay: bug graph-name fallback truoc day ton tai am tham vi
luc do chi test 1 model (traffic_qos_model). Neu chi co 1 model trong bo
test, mot thay doi code lam hong model THU HAI (nhung khong dung lam bo
test) se khong bi phat hien. Co it nhat 2 model KHAC NHAU ro ret (classifier
nho vs detector lon) giup bat duoc loai bug chi lo ra o 1 trong 2 dang do.

Cach dung:
    python3 tools/regression_test.py
    python3 tools/regression_test.py --only traffic_qos_model
    python3 tools/regression_test.py --qairt-root /path/to/qairt/2.44.0.260225

Them model moi vao bo test: sua MODEL_CONFIGS ben duoi, KHONG can sua logic
chay o duoi file.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_TOOL = REPO_ROOT / 'tools' / 'model_deploy.py'

# --------------------------------------------------------------------------
# Danh sach model trong bo regression test. Moi entry la 1 dict tham so
# truyen thang cho `model_deploy.py deploy`. Sua doi tai day khi:
#   - them model moi vao bo test
#   - doi duong dan model/input/reference tren may (vi du chuyen sang may
#     khac, hoac doi vi tri luu model_test/)
#
# LUU Y: cac duong dan ben duoi la placeholder theo dung quy uoc da dung
# trong scripts/*.sh (thu muc /home/congtuan/model_test/). Sua lai cho
# khop thuc te truoc khi chay - script se bao loi ro rang neu file khong
# ton tai, khong chay ngam.
MODEL_CONFIGS = [
    {
        'name': 'traffic_qos_model',
        'model': '/home/congtuan/model_test/traffic_qos_model.onnx',
        'input_raw': '/home/congtuan/model_test/input_seq.raw',
        # Khop dung file reference da dung trong vi du 'validate' o docs/part16
        # (onnx_output_ref.npy) - .npy duoc ho tro thang bang numpy trong
        # model_deploy.py, khong can convert sang .raw.
        'reference': '/home/congtuan/model_test/onnx_output_ref.npy',
        'backend': 'cpu',
        'tolerance': 1e-5,
        'source_model_input_shape': None,  # vi du [('input_seq', '1,10,19')] neu can
        'out_tensor_node': None,
    },
    {
        'name': 'best_onnx_yolo',
        'model': '/home/congtuan/model_test/best.onnx',
        'input_raw': '/home/congtuan/model_test/yolo_input.raw',
        'reference': '/home/congtuan/model_test/yolo_reference_output.raw',
        'backend': 'cpu',
        # YOLO co 1.23M param / 151.200 so output - sai so tich luy qua cac
        # lop se lon hon model nho, nen tolerance long hon la hop ly, khong
        # phai dau hieu loi. Neu can chinh, doi so nay thay vi ha thap chuan
        # chung cho ca 2 model.
        'tolerance': 1e-3,
        'source_model_input_shape': None,
        'out_tensor_node': None,
    },
]


def run_deploy(qairt_root: str, cfg: dict, work_root: Path) -> dict:
    work_dir = work_root / cfg['name']
    cmd = [
        sys.executable, str(DEPLOY_TOOL),
        '--qairt-root', qairt_root,
        'deploy',
        '--model', cfg['model'],
        '--input-raw', cfg['input_raw'],
        '--reference', cfg['reference'],
        '--backend', cfg['backend'],
        '--tolerance', str(cfg['tolerance']),
        '--work-dir', str(work_dir),
    ]
    if cfg.get('source_model_input_shape'):
        for name, dims in cfg['source_model_input_shape']:
            cmd.extend(['--source-model-input-shape', name, dims])
    if cfg.get('out_tensor_node'):
        for out_name in cfg['out_tensor_node']:
            cmd.extend(['--out-tensor-node', out_name])

    print(f"\n=== {cfg['name']} ===")
    print('$ ' + ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    # deploy command exit code = 2 khi validation that bai (co chu dich,
    # xem command_deploy trong model_deploy.py) - khong coi la crash, van
    # phai parse JSON in ra de lay chi tiet max_abs_diff/argmax.
    try:
        # JSON la khoi cuoi cung in ra stdout (co the co log truoc do tu
        # cac buoc convert/build-context/run-api).
        json_start = proc.stdout.rindex('{')
        result = json.loads(proc.stdout[json_start:])
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            'name': cfg['name'],
            'validation_passed': False,
            'error': f'could not parse JSON output from deploy: {exc}',
            'returncode': proc.returncode,
        }

    result['name'] = cfg['name']
    result['returncode'] = proc.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--qairt-root', default='/home/congtuan/qairt_sdk/qairt/2.44.0.260225')
    parser.add_argument('--work-root', default='/tmp/ai_runtime_regression')
    parser.add_argument('--only', default=None,
                         help='Chi chay 1 model theo ten (vd traffic_qos_model), thay vi ca bo.')
    args = parser.parse_args()

    configs = MODEL_CONFIGS
    if args.only:
        configs = [c for c in MODEL_CONFIGS if c['name'] == args.only]
        if not configs:
            print(f"ERROR: unknown model name '{args.only}'. "
                  f"Available: {[c['name'] for c in MODEL_CONFIGS]}", file=sys.stderr)
            raise SystemExit(1)

    work_root = Path(args.work_root)
    results = [run_deploy(args.qairt_root, cfg, work_root) for cfg in configs]

    print('\n=== Regression Summary ===')
    all_passed = True
    for r in results:
        passed = r.get('validation_passed', False)
        all_passed = all_passed and passed
        status = 'PASS' if passed else 'FAIL'
        detail = ''
        if not passed:
            detail = f" ({r.get('error') or ('max_abs_diff=' + str(r.get('max_abs_diff')))})"
        latency = r.get('latency_ms')
        latency_str = f", latency={latency:.3f}ms" if latency is not None else ''
        print(f"  [{status}] {r['name']}{latency_str}{detail}")

    if not all_passed:
        print('\nRegression FAILED - see details above.', file=sys.stderr)
        raise SystemExit(1)

    print('\nAll models passed regression.')


if __name__ == '__main__':
    main()
