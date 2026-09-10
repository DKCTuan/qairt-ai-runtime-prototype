"""Trusted loader for the TinyGRU P16 final training checkpoint.

Use with ``modeltool --model-loader``. The architecture dimensions are
derived from the saved weights; only the training-only ``pair_classifier``
head is intentionally omitted from the inference graph.
"""


def load_model(checkpoint_path: str):
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")

    required = (
        "p16_embedding.weight",
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "classifier.0.weight",
        "classifier.1.weight",
        "classifier.4.weight",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError("checkpoint is missing: " + ", ".join(missing))

    vocab_size, embedding_dim = state["p16_embedding.weight"].shape
    hidden_size = state["gru.weight_hh_l0"].shape[1]
    combined_input_size = state["gru.weight_ih_l0"].shape[1]
    numeric_size = combined_input_size - embedding_dim
    num_classes = state["classifier.4.weight"].shape[0]
    num_layers = len([
        name for name in state
        if name.startswith("gru.weight_ih_l") and "reverse" not in name
    ])
    if numeric_size <= 0 or num_layers <= 0:
        raise ValueError("invalid GRU dimensions in checkpoint")

    class TinyGRUP16Inference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p16_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.gru = torch.nn.GRU(
                input_size=combined_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=0.0,
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(hidden_size, num_classes),
            )

        def forward(self, numeric, p16_ids):
            embedded = self.p16_embedding(p16_ids)
            sequence = torch.cat((numeric, embedded), dim=-1)
            _, hidden = self.gru(sequence)
            return self.classifier(hidden[-1])

    model = TinyGRUP16Inference()
    inference_state = {
        name: value for name, value in state.items()
        if not name.startswith("pair_classifier.")
    }
    result = model.load_state_dict(inference_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "checkpoint does not match inferred TinyGRU P16 architecture; missing="
            f"{result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.eval()
    return model
