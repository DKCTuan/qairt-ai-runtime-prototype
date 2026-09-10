# Stable runtime and model packages

Traffic AI uses one stable C runtime and build-selected model profiles. A
model profile records ordered input/output tensor contracts, class order, and
the preprocessing adapter that turns application data into those tensors.
Model revisions do not need permanent Git branches.

## What changes for each model

- New weights, same contract: replace the generated model library and update
  `model_id` in a copied profile.
- New tensor shape/dtype/count: create a new profile and verify the matching
  adapter supports it.
- New feature semantics: implement a new preprocessing adapter once and name
  it in the profile.

The runtime validates the profile against the real generated model ABI during
initialization. A mismatch fails closed instead of executing with incorrectly
ordered or sized tensors.

## Current adapter

`tinygru_p16_v1` accepts raw/directional packet windows and produces the two
TinyGRU P16 inputs. It remains source-compatible with the package already
validated on the OpenWrt musl target.
