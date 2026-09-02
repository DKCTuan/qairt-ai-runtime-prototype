#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import model_deploy


def normalize_body(body):
    return body if isinstance(body, dict) else {}


def namespace_for_run(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        dlc=body['dlc'],
        input_raw=body['input_raw'],
        input_name=body.get('input_name', 'input_seq'),
        output_name=body.get('output_name', 'class_probs'),
        output_dir=body.get('output_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_run_output')),
        work_dir=body.get('work_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_work')),
        log_level=body.get('log_level', 'error'),
    )


def namespace_for_validate(body):
    ns = namespace_for_run(body)
    ns.reference = body['reference']
    ns.tolerance = float(body.get('tolerance', 1e-5))
    ns.relative_tolerance = float(body.get('relative_tolerance', 0.0))
    ns.output_dtype = body.get('output_dtype', 'float32')
    ns.output_scale = float(body.get('output_scale', 1.0))
    ns.output_zero_point = int(body.get('output_zero_point', 0))
    ns.reference_dtype = body.get('reference_dtype', 'float32')
    ns.reference_scale = float(body.get('reference_scale', 1.0))
    ns.reference_zero_point = int(body.get('reference_zero_point', 0))
    ns.require_argmax = bool(body.get('require_argmax', True))
    return ns


def namespace_for_convert(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        model=body['model'],
        output=body.get('output'),
        converter=body.get('converter', 'auto'),
        source_model_input_shape=body.get('source_model_input_shape'),
        pytorch_input_dim=body.get('pytorch_input_dim'),
        out_tensor_node=body.get('out_tensor_node'),
        extra_args=body.get('extra_args', []),
    )


def namespace_for_prepare(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        model=body['model'],
        model_positional=None,
        work_dir=body.get('work_dir'),
        converter=body.get('converter', 'auto'),
        source_model_input_shape=body.get('source_model_input_shape'),
        pytorch_input_dim=body.get('pytorch_input_dim'),
        out_tensor_node=body.get('out_tensor_node'),
        extra_args=body.get('extra_args', []),
        backend=body.get('backend', 'cpu'),
        target=body.get('target', model_deploy.DEFAULT_TARGET),
    )


def namespace_for_deploy(body):
    """Khop dung tham so command_deploy() dang doi trong model_deploy.py -
    day la duong CHINH (convert -> build-context -> run-api -> so sanh
    reference neu co), khong con qua qnn-net-run nhu /run-host, /validate.
    Truoc ban vá nay, REST API "cham" hon CLI 1 buoc (xem docs Phan 17) -
    vá nay dua REST API theo kip CLI."""
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        target=body.get('target', 'x86_64-linux-clang'),
        model=body['model'],
        work_dir=body.get('work_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_deploy')),
        converter=body.get('converter', 'auto'),
        source_model_input_shape=body.get('source_model_input_shape'),
        pytorch_input_dim=body.get('pytorch_input_dim'),
        out_tensor_node=body.get('out_tensor_node'),
        extra_args=body.get('extra_args', []),
        backend=body.get('backend', 'cpu'),
        input_raw=body['input_raw'],
        graph_name=body.get('graph_name'),
        cc=body.get('cc'),
        ar=body.get('ar'),
        reference=body.get('reference'),
        tolerance=float(body.get('tolerance', 1e-5)),
    )


def namespace_for_doctor(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        model=body.get('model'),
        backend=body.get('backend', 'cpu'),
        target=body.get('target', model_deploy.DEFAULT_TARGET),
        htp_arch=body.get('htp_arch'),
    )


def namespace_for_inspect_context(body):
    body = normalize_body(body)
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        context=body['context'], backend=body.get('backend', 'cpu'),
        target=body.get('target', model_deploy.DEFAULT_TARGET),
        graph_name=body.get('graph_name'),
        work_dir=body.get('work_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_inspect_context')),
    )


def namespace_for_run_context(body):
    body = normalize_body(body)
    raw_inputs = body['input_raw']
    if isinstance(raw_inputs, str):
        raw_inputs = [raw_inputs]
    return argparse.Namespace(
        qairt_root=body.get('qairt_root', str(model_deploy.DEFAULT_QAIRT_ROOT)),
        context=body['context'], input_raw=raw_inputs,
        output_dir=body.get('output_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_run_context_output')),
        backend=body.get('backend', 'cpu'),
        target=body.get('target', model_deploy.DEFAULT_TARGET),
        graph_name=body.get('graph_name'),
        work_dir=body.get('work_dir', str(model_deploy.DEFAULT_WORK_ROOT / 'api_run_context')),
    )


def capture_json(func, args):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        func(args)
    text = stream.getvalue()
    return _parse_last_json(text)


def capture_json_allow_exit(func, args):
    """Giong capture_json, nhung KHONG de mat JSON da in ra neu func()
    ket thuc bang raise SystemExit (vd command_deploy khi validation that
    bai: in JSON chi tiet ROI MOI raise SystemExit(2) - neu khong bat o
    day, JSON do se bi nuot mat, /deploy chi tra ve loi 500 chung chung
    thay vi max_abs_diff/argmax that."""
    stream = io.StringIO()
    exit_code = 0
    try:
        with contextlib.redirect_stdout(stream):
            func(args)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    text = stream.getvalue()
    return _parse_last_json(text), exit_code


def _parse_last_json(text):
    json_objects = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find('{', index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            break
        json_objects.append(obj)
        index = start + end
    return json_objects[-1] if json_objects else {'status': 'success', 'log': text}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        data = json.dumps(payload, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length) if length else b'{}'
        return json.loads(raw.decode('utf-8'))

    def do_GET(self):
        if self.path == '/health':
            self._send(200, {'status': 'ok'})
            return
        self._send(404, {'status': 'error', 'message': 'not found'})

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == '/run-host':
                result = capture_json(model_deploy.command_run_host, namespace_for_run(body))
                self._send(200, result)
            elif self.path == '/validate':
                result = capture_json(model_deploy.command_validate, namespace_for_validate(body))
                status = 200 if result.get('validation_passed') else 422
                self._send(status, result)
            elif self.path == '/convert':
                result = capture_json(model_deploy.command_convert, namespace_for_convert(body))
                self._send(200, result)
            elif self.path == '/doctor':
                result, exit_code = capture_json_allow_exit(model_deploy.command_doctor, namespace_for_doctor(body))
                self._send(200 if exit_code == 0 else 422, result)
            elif self.path == '/inspect-context':
                result = capture_json(model_deploy.command_inspect_context,
                                      namespace_for_inspect_context(body))
                self._send(200, result)
            elif self.path == '/run-context':
                result = capture_json(model_deploy.command_run_context,
                                      namespace_for_run_context(body))
                self._send(200, result)
            elif self.path == '/prepare':
                result = capture_json(model_deploy.command_prepare, namespace_for_prepare(body))
                self._send(200, result)
            elif self.path == '/deploy':
                result, exit_code = capture_json_allow_exit(model_deploy.command_deploy, namespace_for_deploy(body))
                # exit_code 2 = validation that bai co chu dich (co --reference,
                # max_abs_diff/argmax khong khop) - van la ket qua HOP LE, tra
                # 422 kem JSON chi tiet, KHONG phai loi server (5xx).
                # exit_code != 0 va != 2 = loi that su o mot buoc trong chain
                # (convert/build-context/run-api) - tra 500.
                if exit_code == 0:
                    status = 200
                elif exit_code == 2:
                    status = 422
                else:
                    status = 500
                self._send(status, result)
            else:
                self._send(404, {'status': 'error', 'message': 'not found'})
        except KeyError as exc:
            self._send(400, {'status': 'error', 'message': f'missing field: {exc.args[0]}'})
        except SystemExit as exc:
            self._send(500, {'status': 'error', 'message': f'command failed with code {exc.code}'})
        except Exception as exc:
            self._send(500, {'status': 'error', 'message': str(exc)})

    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description='REST API server for model deploy prototype')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()

    # Conversion and runtime commands share default work directories. A serial
    # server avoids two requests overwriting the same logs/artifacts; callers
    # that need parallelism should run separate instances with separate roots.
    server = HTTPServer((args.host, args.port), Handler)
    print(f'listening on http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

