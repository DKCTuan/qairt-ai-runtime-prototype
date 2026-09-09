#!/usr/bin/env python3
"""Create one raw TinyGRU input tensor from a packet CSV.

The default output is exactly 270 little-endian float32 values: 90 packet-major
raw triples [delta_time, direction, packet_length]. It is accepted by
``app_arm64 gru INPUT.raw``; the app/API performs model normalization exactly
once. Use --normalized only with an explicit normalized-input API/mode.

This helper supports the two CSV header styles present in the supplied traffic
classification data.  It is a reproducible integration-test extractor, not a
replacement for the production flow tracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path


HEADER_ALIASES = {
    "time": ("Time", "frame.time_epoch"),
    "source": ("Source", "ip.src"),
    "destination": ("Destination", "ip.dst"),
    "length": ("Length", "frame.len"),
}


def pick(row: dict[str, str], name: str) -> str:
    for column in HEADER_ALIASES[name]:
        if column in row and row[column] not in (None, ""):
            return row[column]
    raise ValueError(f"CSV has no usable {name} column; expected one of {HEADER_ALIASES[name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV packet trace -> TinyGRU .raw input")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--local-ip", required=True,
                        help="IP of the endpoint whose outbound packets receive +1 direction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path,
                        default=Path("traffic_models/vendor/preprocessing_metadata.json"))
    parser.add_argument("--outbound-direction", type=float, default=1.0,
                        help="direction value for source == --local-ip (default: 1.0)")
    parser.add_argument("--normalized", action="store_true",
                        help="write model-normalized values; do not use with app_arm64 gru")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    shape = metadata.get("input_shape")
    if shape != [1, 90, 3] or metadata.get("input_dtype") != "float32":
        raise SystemExit(f"unsupported TinyGRU contract in {args.metadata}: {shape}")
    mean = metadata["continuous_mean"]
    std = metadata["continuous_std"]
    if len(mean) != 2 or len(std) != 2 or not all(value > 0 for value in std):
        raise SystemExit("invalid continuous normalization metadata")

    packets: list[tuple[float, str, str, float]] = []
    with args.csv_file.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                item = (float(pick(row, "time")), pick(row, "source"),
                        pick(row, "destination"), float(pick(row, "length")))
            except (ValueError, TypeError) as error:
                raise SystemExit(f"invalid packet row {len(packets) + 2}: {error}") from error
            if not all(math.isfinite(value) for value in (item[0], item[3])):
                raise SystemExit(f"non-finite value in packet row {len(packets) + 2}")
            packets.append(item)
            if len(packets) == 90:
                break
    if len(packets) != 90:
        raise SystemExit(f"need at least 90 packets, found {len(packets)}")

    values: list[float] = []
    previous_time = packets[0][0]
    for index, (timestamp, source, _destination, length) in enumerate(packets):
        delta_time = 0.0 if index == 0 else timestamp - previous_time
        previous_time = timestamp
        if delta_time < 0:
            raise SystemExit("packet timestamps must be non-decreasing")
        direction = args.outbound_direction if source == args.local_ip else -args.outbound_direction
        if args.normalized:
            values.extend(((delta_time - mean[0]) / std[0], direction,
                           (length - mean[1]) / std[1]))
        else:
            values.extend((delta_time, direction, length))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(struct.pack("<270f", *values))
    print(f"input={args.csv_file}")
    print(f"output={args.output}")
    print("shape=[1,90,3] dtype=float32 bytes=1080 normalized=" + ("yes" if args.normalized else "no"))
    print(f"direction: source == {args.local_ip} -> {args.outbound_direction:g}; otherwise -> {-args.outbound_direction:g}")


if __name__ == "__main__":
    main()
