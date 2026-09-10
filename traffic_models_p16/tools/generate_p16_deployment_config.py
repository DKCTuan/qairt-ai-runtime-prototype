#!/usr/bin/env python3
"""Generate a static TinyGRU P16 deployment config from training artifacts."""

import argparse
import csv
import ipaddress
import json
import math
from pathlib import Path


WINDOW = 90
CHANNELS = (
    "log1p_packet_length",
    "log1p_nonnegative_iat",
    "direction",
)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def finite_float(value: object, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{description} is not a number")
    if not math.isfinite(result):
        fail(f"{description} is not finite")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("window_size") != WINDOW or metadata.get("stride") != WINDOW:
        fail("metadata must specify window_size=stride=90")
    if metadata.get("skip_packets") != 10:
        fail("metadata must specify skip_packets=10")
    if metadata.get("input_numeric_shape") != [1, WINDOW, 3]:
        fail("input_numeric_shape must be [1,90,3]")
    if metadata.get("input_p16_shape") != [1, WINDOW]:
        fail("input_p16_shape must be [1,90]")
    if metadata.get("numeric_features") != [
        "log1p(packet_length)", "log1p(nonnegative_iat)", "direction",
    ]:
        fail("numeric feature order differs from the P16 deployment contract")
    classes = metadata.get("classes")
    if not isinstance(classes, list) or len(classes) != 5 or not all(isinstance(x, str) for x in classes):
        fail("metadata must contain exactly five class names")
    model_profile_path = args.output_dir / "model_profile.json"
    if model_profile_path.is_file():
        try:
            model_profile = json.loads(model_profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            fail(f"cannot read matching model profile: {error}")
        if model_profile.get("preprocessing") != "tinygru_p16_v1":
            fail("model profile does not select the tinygru_p16_v1 adapter")
        if model_profile.get("classes") != classes:
            fail("model profile class order differs from training metadata")
        if model_profile.get("inputs") != [
            {"name": "numeric", "dtype": "float32", "shape": [1, WINDOW, 3]},
            {"name": "p16_ids", "dtype": "int32", "shape": [1, WINDOW]},
        ]:
            fail("model profile inputs differ from the P16 deployment contract")
        if model_profile.get("outputs") != [
            {"name": "scores", "dtype": "float32", "shape": [1, 5]},
        ]:
            fail("model profile output differs from the P16 deployment contract")
    p16 = metadata.get("p16", {})
    if p16.get("unk_id") != 0 or p16.get("embedding_dim") != 8:
        fail("P16 metadata must use embedding_dim=8 and unk_id=0")
    vocab_size = p16.get("vocab_size_including_unk")
    if not isinstance(vocab_size, int) or vocab_size < 1:
        fail("invalid p16 vocabulary size")

    rows = list(csv.DictReader(args.normalization.open(encoding="utf-8", newline="")))
    if [row.get("feature") for row in rows] != list(CHANNELS):
        fail("normalization.csv channels are missing or in the wrong order")
    means = [finite_float(row.get("mean"), f"mean[{i}]") for i, row in enumerate(rows)]
    stds = [finite_float(row.get("std"), f"std[{i}]") for i, row in enumerate(rows)]
    if any(x <= 0.0 for x in stds):
        fail("normalization standard deviations must be positive")
    meta_norm = metadata.get("normalization", {})
    if "mean" in meta_norm and "std" in meta_norm:
        for actual, expected in zip(means, meta_norm["mean"]):
            if not math.isclose(actual, finite_float(expected, "metadata mean"), rel_tol=1e-6, abs_tol=1e-7):
                fail("normalization.csv mean differs from metadata")
        for actual, expected in zip(stds, meta_norm["std"]):
            if not math.isclose(actual, finite_float(expected, "metadata std"), rel_tol=1e-6, abs_tol=1e-7):
                fail("normalization.csv std differs from metadata")

    entries = []
    seen_prefixes, seen_ids = set(), set()
    for row in csv.DictReader(args.vocabulary.open(encoding="utf-8", newline="")):
        try:
            network = ipaddress.ip_network(row["prefix"], strict=True)
            embedding_id = int(row["embedding_id"])
        except (KeyError, ValueError) as exc:
            fail(f"invalid vocabulary row: {exc}")
        if network.version != 4 or network.prefixlen != 16:
            fail(f"vocabulary prefix is not IPv4 /16: {network}")
        prefix = int(network.network_address) >> 16
        if prefix in seen_prefixes or embedding_id in seen_ids:
            fail("vocabulary contains duplicate prefix or embedding ID")
        if embedding_id <= 0 or embedding_id >= vocab_size:
            fail(f"embedding ID {embedding_id} is outside 1..{vocab_size - 1}")
        seen_prefixes.add(prefix)
        seen_ids.add(embedding_id)
        entries.append((prefix, embedding_id))
    if len(entries) != vocab_size - 1 or seen_ids != set(range(1, vocab_size)):
        fail("vocabulary must contain every embedding ID exactly once")
    entries.sort()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    header = output / "traffic_p16_deployment_config.h"
    source = output / "traffic_p16_deployment_config.c"
    header.write_text(
        "#ifndef TRAFFIC_P16_DEPLOYMENT_CONFIG_H\n"
        "#define TRAFFIC_P16_DEPLOYMENT_CONFIG_H\n\n"
        "#include \"traffic_p16.h\"\n\n"
        "const traffic_p16_config_t *traffic_p16_default_config(void);\n"
        "const char *traffic_p16_class_name(int label);\n\n"
        "#endif\n", encoding="utf-8")
    entry_text = ",\n".join(f"    {{0x{prefix:04x}U, {ident}}}" for prefix, ident in entries)
    source.write_text(
        "/* Generated by tools/generate_p16_deployment_config.py; do not edit. */\n"
        "#include \"traffic_p16_deployment_config.h\"\n\n"
        "static const traffic_p16_vocab_entry_t vocabulary[] = {\n" + entry_text + "\n};\n\n"
        "static const traffic_p16_config_t config = {\n"
        f"    {{{means[0]:.9g}f, {means[1]:.9g}f, {means[2]:.9g}f}},\n"
        f"    {{{stds[0]:.9g}f, {stds[1]:.9g}f, {stds[2]:.9g}f}},\n"
        "    vocabulary, sizeof(vocabulary) / sizeof(vocabulary[0]), " + str(vocab_size) + "\n};\n\n"
        "const traffic_p16_config_t *traffic_p16_default_config(void) { return &config; }\n",
        encoding="utf-8")
    print(f"status=success\nentries={len(entries)}\nvocab_size={vocab_size}\noutput={output}")


if __name__ == "__main__":
    main()
