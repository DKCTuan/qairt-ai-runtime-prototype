"""Template supplied by the model owner for a state_dict-only .pth checkpoint.

Copy this file next to the training model code, implement one of the two
interfaces below, and pass its path using ``--model-loader``.  It is executed
on the host during conversion, never copied to the ARM64 target.
"""

# Option A (recommended): build the model and load the checkpoint yourself.
# This supports any checkpoint naming/layout used by the training project.
#
# from my_training_model import TrafficClassifier
# import torch
#
# def load_model(checkpoint_path: str):
#     model = TrafficClassifier(input_size=..., num_classes=...)
#     checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
#     state = checkpoint.get("state_dict", checkpoint)
#     model.load_state_dict(state)
#     return model.eval()


# Option B: return only the architecture.  modeltool then loads a plain
# state_dict, model_state_dict, or state_dict checkpoint field automatically.
#
# from my_training_model import TrafficClassifier
#
# def build_model():
#     return TrafficClassifier(input_size=..., num_classes=...)
