# TinyGRU P16 H48x2 cap-0.25 profile

This profile implements the deployment contract emitted by
`wifi-qos_TinyGRU_FINAL_train_save_H48x2.ipynb`.

- Numeric input: `float32 [1,90,3]` in the order
  `log1p(packet_length)`, `log1p(max(IAT_seconds,0))`, `direction`, then
  z-score all three channels with this profile's constants.
- P16 input: `int32 [1,90]`; global-unicast remote IPv4 `/16` vocabulary ID,
  with invalid, non-global, and unseen prefixes mapped to `0` (UNK).
- Direction: `+1` local/LAN to remote/WAN, `-1` remote/WAN to local/LAN.
- Windowing: use packets 1..90 (`skip_packets=0`, `stride=90`).
- Output: raw logits in `Background, Game, RTVideo, Voice, VStream` order.
  The SDK computes softmax at temperature 1 and marks a result accepted when
  confidence is at least `0.98`.

Architecture: a two-layer, unidirectional GRU with numeric input size 3 and
hidden size 48. A two-dimensional P16 embedding is mean-pooled over non-UNK
packets and late-fused into the five traffic logits through
`0.25 * tanh(ip_to_logits(...))`.

Source provenance (SHA-256):

- bundle: `7436ea39dc305c412ceb855ef218ab182fbbd8192b3aefc933ccec5368397bd0`
- training notebook: `804bbe263dd3b2c6e735e7aedee11d08bf524696d76e524f8e5154d6530059a1`
- inference checkpoint: `81beda109f6a2a5f4b43068c7fbf84243cf3aca5164edec579eb76e0b6c7b422`
- training checkpoint: `1758dd0f3a50fa6a9919cf566a23919759c599a2faaf0ad58d14ad1e88c8dabf`
