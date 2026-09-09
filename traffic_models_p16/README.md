# TinyGRU P16 integration (separate from V6)

This directory is intentionally separate from `traffic_models/`: the V6 SDK
uses a one-input TinyGRU and must not be linked with this two-input model.

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
