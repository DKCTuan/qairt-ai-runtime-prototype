# Model profiles

A profile is the build-time contract between a converted model library and
the stable Traffic AI runtime. Select it without creating a Git branch:

```bash
make MODEL_PROFILE=model_profiles/tinygru_p16/model_profile.json \
  MODEL_LIBRARY=/path/to/libtiny_gru_p16_float32.so \
  MODEL_INCLUDE=/path/to/generated/model/include
```

The profile declares the ordered tensors, dtypes, static shapes, class order,
and a preprocessing adapter ID. `tools/generate_model_profile.py` validates
it and emits the C contract into the build output. At model initialization,
the runtime compares this contract with the tensors reported by the actual
model library.

Adapter assets live beside `model_profile.json`. The P16 profile therefore
also owns `traffic_p16_deployment_config.c/.h`, generated from the matching
normalization and vocabulary sidecars. This prevents ABI metadata from one
model being built with preprocessing data from another model.

Changing weights while keeping the same tensor and preprocessing contract
only requires a new model library and a new `model_id`. Changing tensor
shapes/dtypes requires a new profile. A new preprocessing meaning requires a
new adapter implementation, but it still belongs in this directory structure
and does not require a permanent Git branch.

Currently supported production adapter:

- `tinygru_p16_v1`: numeric float32 `[1,90,3]`, P16 IDs int32 `[1,90]`,
  scores float32 `[1,5]`.
