# Unified Random Forest + TinyGRU P16 SDK

This directory produces the current unified SDK: Random Forest is compiled
directly into `libtraffic_ai.so`, while TinyGRU P16 remains its own model
library (`libtiny_gru_p16_float32.so`). The legacy one-input TinyGRU is not
part of this build.

`traffic_p16_prepare_packets()` receives the 90 post-skip packets and creates
both TFLite tensors itself. Every record must contain a microsecond timestamp,
the training-aligned `Length` value, source IPv4 and destination IPv4. A
numeric `[90][3]` vector alone is insufficient because it cannot produce the
model's `/16` embedding IDs.

Before a production build, generate a static `traffic_p16_config_t` from the
same final checkpoint as `tiny_gru_p16_float32.tflite`:

- normalization mean/std for all three numeric channels;
- the sorted `/16` to embedding-ID vocabulary;
- class labels in model output order.

Use the generator to turn the three sidecars into source compiled into the
SDK; the target then has no CSV/JSON parsing dependency:

```bash
python3 tools/generate_p16_deployment_config.py \
  --metadata model_metadata.json \
  --normalization normalization.csv \
  --vocabulary p16_vocabulary.csv \
  --output-dir generated
```

The model ABI is verified at initialization. It accepts only TFLite input 0
`float32 [1,90,3]`, input 1 `int32 [1,90]`, and output `float32 [1,5]`.

To build the unified SDK alongside its generated P16 model library:

```bash
make OUT=dist/traffic_ai \
  GRU_LIBRARY=dist/tiny_gru_p16/arm64/libtiny_gru_p16_float32.so \
  GRU_INCLUDE=dist/tiny_gru_p16/arm64 \
  CC=aarch64-openwrt-linux-musl-gcc \
  STRIP=aarch64-openwrt-linux-musl-strip
```

The build produces `app_arm64`, a deterministic smoke client. It
exercises the full raw-packet preprocessing path and is not a substitute for
parity validation using a real captured window.

For handoff, use
`tools/package_combined_musl.sh`. The resulting archive has a `bin/` + `lib/`
layout and includes `docs/HANDOFF.md`.
