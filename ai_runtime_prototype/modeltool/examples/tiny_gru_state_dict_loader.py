"""Reference loader for the supplied TinyGRU weights-only checkpoint.

It demonstrates the generic ``build_model()`` contract.  Other PyTorch
projects should supply their own loader rather than adding model-specific code
to modeltool itself.
"""

import torch


class TinyGRUInference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=3,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 5),
        )

    def forward(self, value):
        _, hidden = self.gru(value)
        return self.classifier(hidden[-1])


def load_model(checkpoint_path: str):
    """Handle this training checkpoint's auxiliary pair-classifier head."""
    model = TinyGRUInference()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    inference_state = {
        name: value for name, value in state.items()
        if not name.startswith("pair_classifier.")
    }
    model.load_state_dict(inference_state, strict=True)
    return model.eval()
