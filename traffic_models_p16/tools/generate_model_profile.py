#!/usr/bin/env python3
"""Validate a deployment profile and generate its small C ABI contract."""

import argparse
import json
import math
from pathlib import Path


DTYPES = {"float32": "TRAFFIC_MODEL_DTYPE_FLOAT32", "int32": "TRAFFIC_MODEL_DTYPE_INT32"}


def fail(message):
    raise SystemExit("error: " + message)


def c_string(value):
    return json.dumps(value, ensure_ascii=True)


def c_float(value):
    text = format(float(value), ".9g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def tensor(item, location):
    if not isinstance(item, dict):
        fail(location + " must be an object")
    name, dtype, shape = item.get("name"), item.get("dtype"), item.get("shape")
    if not isinstance(name, str) or not name:
        fail(location + ".name must be a non-empty string")
    if dtype not in DTYPES:
        fail(location + ".dtype must be one of: " + ", ".join(sorted(DTYPES)))
    if not isinstance(shape, list) or not shape or not all(isinstance(x, int) and x > 0 for x in shape):
        fail(location + ".shape must contain positive static dimensions")
    elements = math.prod(shape)
    return name, DTYPES[dtype], elements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        fail("cannot read profile: " + str(error))
    if profile.get("schema_version") != 1:
        fail("schema_version must be 1")
    model_id = profile.get("model_id")
    preprocessing = profile.get("preprocessing")
    if not isinstance(model_id, str) or not model_id:
        fail("model_id must be a non-empty string")
    if not isinstance(preprocessing, str) or not preprocessing:
        fail("preprocessing must be a non-empty adapter ID")
    raw_inputs, raw_outputs = profile.get("inputs"), profile.get("outputs")
    if not isinstance(raw_inputs, list) or not 1 <= len(raw_inputs) <= 8:
        fail("inputs must contain 1..8 tensors")
    if not isinstance(raw_outputs, list) or not 1 <= len(raw_outputs) <= 8:
        fail("outputs must contain 1..8 tensors")
    inputs = [tensor(value, "inputs[%d]" % i) for i, value in enumerate(raw_inputs)]
    outputs = [tensor(value, "outputs[%d]" % i) for i, value in enumerate(raw_outputs)]
    classes = profile.get("classes", [])
    if not isinstance(classes, list) or not all(isinstance(x, str) and x for x in classes):
        fail("classes must be an array of non-empty strings")
    if classes and outputs[0][2] != len(classes):
        fail("the first output element count must match classes")
    windowing = profile.get("windowing", {})
    skip_packets = windowing.get("skip_packets", 0)
    if not isinstance(skip_packets, int) or skip_packets < 0:
        fail("windowing.skip_packets must be a non-negative integer")
    postprocessing = profile.get("postprocessing", {})
    temperature = postprocessing.get("softmax_temperature", 1.0)
    threshold = postprocessing.get("accept_threshold", 0.0)
    if (not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
            or temperature <= 0.0):
        fail("postprocessing.softmax_temperature must be positive and finite")
    if (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or threshold < 0.0 or threshold > 1.0):
        fail("postprocessing.accept_threshold must be in [0,1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = args.output_dir / "traffic_model_profile.h"
    source = args.output_dir / "traffic_model_profile.c"
    header.write_text(
        "#ifndef TRAFFIC_MODEL_PROFILE_H\n#define TRAFFIC_MODEL_PROFILE_H\n\n"
        "#include \"model_contract.h\"\n\n"
        "extern const traffic_model_contract_t traffic_model_profile_contract;\n\n"
        "#endif\n", encoding="utf-8")

    def specs(name, values):
        body = ",\n".join("    {%s, %s, %d}" % (c_string(n), d, count)
                            for n, d, count in values)
        return "static const traffic_model_tensor_spec_t %s[] = {\n%s\n};\n\n" % (name, body)

    class_values = classes or ["Unknown"]
    source.write_text(
        "/* Generated from %s; do not edit. */\n" % args.profile.name +
        "#include \"traffic_model_profile.h\"\n\n" +
        specs("profile_inputs", inputs) + specs("profile_outputs", outputs) +
        "static const char *const profile_classes[] = {%s};\n\n" %
        ", ".join(c_string(x) for x in class_values) +
        "const traffic_model_contract_t traffic_model_profile_contract = {\n"
        "    %s, %s,\n" % (c_string(model_id), c_string(preprocessing)) +
        "    profile_inputs, %d, profile_outputs, %d,\n" % (len(inputs), len(outputs)) +
        "    profile_classes, %d,\n" % len(classes) +
        "    %d, %s, %s\n};\n" %
        (skip_packets, c_float(temperature), c_float(threshold)),
        encoding="utf-8")
    print("status=success")
    print("model_id=" + model_id)
    print("preprocessing=" + preprocessing)
    print("inputs=%d" % len(inputs))
    print("outputs=%d" % len(outputs))
    print("output=" + str(args.output_dir))


if __name__ == "__main__":
    main()
