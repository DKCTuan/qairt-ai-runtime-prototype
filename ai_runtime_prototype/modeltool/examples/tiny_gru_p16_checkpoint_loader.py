"""Trusted loader for TinyGRU P16 training or inference checkpoints.

Use with ``modeltool --model-loader``. The architecture dimensions are
derived from the saved weights. Both the original early-fusion model and the
H48x2 capped late-fusion model are supported; training-only auxiliary heads
are intentionally omitted from the inference graph.
"""


def load_model(checkpoint_path: str, *, trust_pickle: bool = False):
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        if not trust_pickle:
            raise ValueError(
                "checkpoint cannot be loaded with weights_only=True; pass "
                "--trust-pytorch-pickle only when the checkpoint is trusted"
            ) from error
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")

    late_fusion = "ip_to_logits.weight" in state
    head = "traffic_head" if late_fusion else "classifier"
    required = (
        "p16_embedding.weight", "gru.weight_ih_l0", "gru.weight_hh_l0",
        f"{head}.0.weight", f"{head}.1.weight", f"{head}.4.weight",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError("checkpoint is missing: " + ", ".join(missing))

    vocab_size, embedding_dim = state["p16_embedding.weight"].shape
    hidden_size = state["gru.weight_hh_l0"].shape[1]
    gru_input_size = state["gru.weight_ih_l0"].shape[1]
    numeric_size = gru_input_size if late_fusion else gru_input_size - embedding_dim
    num_classes = state[f"{head}.4.weight"].shape[0]
    num_layers = len([
        name for name in state
        if name.startswith("gru.weight_ih_l") and "reverse" not in name
    ])
    if numeric_size <= 0 or num_layers <= 0:
        raise ValueError("invalid GRU dimensions in checkpoint")

    if late_fusion:
        required_late = ("ip_to_logits.weight",)
        missing_late = [name for name in required_late if name not in state]
        if missing_late:
            raise ValueError("late-fusion checkpoint is missing: " + ", ".join(missing_late))
        config = checkpoint.get("config", {})
        if not isinstance(config, dict) or "ip_logit_cap" not in config:
            raise ValueError("late-fusion checkpoint config must contain ip_logit_cap")
        ip_logit_cap = float(config["ip_logit_cap"])
        if not 0.0 <= ip_logit_cap <= 1.0:
            raise ValueError("ip_logit_cap must be in [0, 1]")
        if state["ip_to_logits.weight"].shape != (num_classes, embedding_dim):
            raise ValueError("ip_to_logits dimensions do not match the checkpoint")

    class TinyGRUP16EarlyFusionInference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p16_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.gru = torch.nn.GRU(
                input_size=gru_input_size,
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

    class TinyGRUP16LateFusionInference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(
                input_size=numeric_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=False, dropout=0.0,
            )
            self.traffic_head = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(), torch.nn.Dropout(0.0),
                torch.nn.Linear(hidden_size, num_classes),
            )
            self.p16_embedding = torch.nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            self.ip_to_logits = torch.nn.Linear(
                embedding_dim, num_classes, bias=False
            )

        def forward(self, numeric, p16_ids):
            _, hidden = self.gru(numeric)
            traffic_logits = self.traffic_head(hidden[-1])
            embedded = self.p16_embedding(p16_ids)
            valid = (p16_ids != 0).unsqueeze(-1).to(embedded.dtype)
            pooled = (embedded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            ip_delta = ip_logit_cap * torch.tanh(self.ip_to_logits(pooled))
            return traffic_logits + ip_delta

    model = (TinyGRUP16LateFusionInference() if late_fusion
             else TinyGRUP16EarlyFusionInference())
    ignored_prefixes = (
        "pair_classifier.", "rtvideo_voice_head.", "bg_vstream_head."
    )
    inference_state = {name: value for name, value in state.items()
                       if not name.startswith(ignored_prefixes)}
    result = model.load_state_dict(inference_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "checkpoint does not match inferred TinyGRU P16 architecture; missing="
            f"{result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.eval()
    return model
