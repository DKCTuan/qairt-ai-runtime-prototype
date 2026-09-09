# TinyGRU P16 model provenance

The generated deployment config in `generated/` was created from the final
training artifact ZIP supplied on 2026-09-09.

| Artifact | SHA-256 |
| --- | --- |
| `tiny_gru_p16_float32.tflite` | `58ed38ed770623a57f7ff706b6438131231a0a619629cee1a3fee1102e3dba43` |
| `tinygru_p16_final_training_checkpoint.pt` | `cbdc73cf33cc2095f34b3c538da770559e8de4176f8df5c615c5378b3c70e6ca` |
| `model_metadata.json` | `345c27bbb9bd4c745dc0853ee6825d23819402849eb58eae4b76ebd2b8db8f08` |
| `normalization.csv` | `ba3c476f6edb9d5d1940786a9d5c2048738922f1adce0596f421b0ff0fc62cc7` |
| `p16_vocabulary.csv` | `e4df36fce6a2d4919981f6bb9c6753bba76edaca672e77b756a88b9328d133a0` |

Deployment contract: 90-packet windows after skipping the first 10 packets,
two inputs `[1,90,3] float32` and `[1,90] int32`, five classes in this order:
`Background`, `Game`, `RTVideo`, `Voice`, `VStream`.
