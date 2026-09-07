#!/usr/bin/env python3
"""Export supported PyTorch checkpoints to a float32 TFLite model.

This helper is intentionally executed with the separate ``model-torch``
environment.  It currently auto-detects the TinyGRU checkpoint layout used by
the traffic model: ``model_config`` plus ``model_state_dict``.  The command
fails with an actionable message for other checkpoint layouts instead of
guessing an architecture or preprocessing contract.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        fail(f"invalid --input-shape {value!r}; use comma-separated positive integers")
        raise error
    if not shape or any(part <= 0 for part in shape):
        fail("--input-shape must contain positive integers")
    return shape


def find_metadata(model_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            fail(f"metadata file not found: {candidate}")
        return candidate
    for name in ("preprocessing_metadata.json", "metadata.json"):
        candidate = model_path.parent / name
        if candidate.is_file():
            return candidate
    return None


def shape_from_metadata(metadata_path: Path | None) -> tuple[int, ...] | None:
    if metadata_path is None:
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shape = metadata.get("input_shape")
        if not isinstance(shape, list) or not shape:
            return None
        if not all(isinstance(part, int) and part > 0 for part in shape):
            fail(f"metadata input_shape must contain positive integers: {metadata_path}")
        return tuple(shape)
    except json.JSONDecodeError:
        fail(f"invalid JSON metadata: {metadata_path}")


def build_tiny_gru(torch, config: dict):
    """Rebuild the known TinyGRU inference graph from its saved config."""
    required = ("input_size", "hidden_size", "num_classes")
    missing = [name for name in required if name not in config]
    if missing:
        fail("TinyGRU model_config is missing: " + ", ".join(missing))

    nn = torch.nn

    class TinyGRUInference(nn.Module):
        def __init__(self):
            super().__init__()
            layers = int(config.get("num_gru_layers", 2))
            dropout = float(config.get("gru_dropout", 0.0)) if layers > 1 else 0.0
            hidden = int(config["hidden_size"])
            self.gru = nn.GRU(
                input_size=int(config["input_size"]),
                hidden_size=hidden,
                num_layers=layers,
                batch_first=bool(config.get("batch_first", True)),
                bidirectional=bool(config.get("bidirectional", False)),
                dropout=dropout,
            )
            output_hidden = hidden * (2 if bool(config.get("bidirectional", False)) else 1)
            self.classifier = nn.Sequential(
                nn.LayerNorm(output_hidden),
                nn.Linear(output_hidden, output_hidden),
                nn.ReLU(),
                nn.Dropout(float(config.get("classifier_dropout", 0.0))),
                nn.Linear(output_hidden, int(config["num_classes"])),
            )

        def forward(self, value):
            _, hidden_state = self.gru(value)
            if self.gru.bidirectional:
                last_hidden = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
            else:
                last_hidden = hidden_state[-1]
            return self.classifier(last_hidden)

    return TinyGRUInference()


def _load_external_model(torch, model_path: Path, loader_path: Path, trust_pickle: bool):
    """Load a user-supplied architecture without hard-coding it in the tool.

    A loader must expose either ``load_model(checkpoint_path)`` returning an
    evaluated ``torch.nn.Module``, or ``build_model()`` returning a module to
    which this helper applies a plain state dict.  Supplying a loader is an
    explicit trust boundary: it is Python code chosen by the model owner.
    """
    if not loader_path.is_file():
        fail(f"model loader not found: {loader_path}")
    spec = importlib.util.spec_from_file_location("modeltool_user_loader", loader_path)
    if spec is None or spec.loader is None:
        fail(f"cannot import model loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    loader_dir = str(loader_path.parent)
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        fail(f"MODEL_LOADER_FAILED: cannot import {loader_path}: {error}")

    load_function = getattr(module, "load_model", None)
    if callable(load_function):
        try:
            model = load_function(str(model_path))
        except Exception as error:
            fail(f"MODEL_LOADER_FAILED: load_model() failed: {error}")
        if not isinstance(model, torch.nn.Module):
            fail("MODEL_LOADER_FAILED: load_model() must return torch.nn.Module")
        model.eval()
        return model, "external-model-loader"

    build_function = getattr(module, "build_model", None)
    if not callable(build_function):
        fail("PYTORCH_MODEL_LOADER_REQUIRED: loader must define "
             "load_model(checkpoint_path) or build_model()")
    try:
        model = build_function()
    except Exception as error:
        fail(f"MODEL_LOADER_FAILED: build_model() failed: {error}")
    if not isinstance(model, torch.nn.Module):
        fail("MODEL_LOADER_FAILED: build_model() must return torch.nn.Module")
    try:
        # A plain state dict can be read without unpickling arbitrary custom
        # Python classes.  Older PyTorch releases lack weights_only, so retain
        # an explicit trusted fallback for those releases/checkpoint layouts.
        try:
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except (TypeError, RuntimeError):
            if not trust_pickle:
                fail("PYTORCH_MODEL_LOADER_REQUIRED: checkpoint needs pickle loading; "
                     "re-run with --trust-pytorch-pickle only if it is trusted")
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except Exception as error:
        fail(f"MODEL_LOADER_FAILED: cannot read checkpoint: {error}")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    if not isinstance(state, dict):
        fail("MODEL_LOADER_FAILED: checkpoint does not contain a state dict")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        fail("MODEL_LOADER_FAILED: state dict does not match model; missing="
             f"{result.missing_keys}, unexpected={result.unexpected_keys}")
    model.eval()
    return model, "external-model-loader"


def load_model(torch, model_path: Path, trust_pickle: bool, loader_path: Path | None):
    """Load TorchScript or the supported TinyGRU checkpoint format."""
    try:
        module = torch.jit.load(str(model_path), map_location="cpu")
        module.eval()
        return module, "torchscript"
    except (RuntimeError, ValueError):
        pass

    if loader_path is not None:
        return _load_external_model(torch, model_path, loader_path, trust_pickle)

    if not trust_pickle:
        fail("PYTORCH_MODEL_LOADER_REQUIRED: file is not TorchScript. Provide "
             "--model-loader for a state_dict checkpoint, or re-run with "
             "--trust-pytorch-pickle only for a trusted full checkpoint.")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        return checkpoint, "pytorch-module"
    if not isinstance(checkpoint, dict):
        fail("unsupported PyTorch file: expected TorchScript, nn.Module, or checkpoint dictionary")

    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if isinstance(config, dict) and isinstance(state, dict):
        model = build_tiny_gru(torch, config)
        # Training checkpoints can contain auxiliary-head weights.  The
        # inference wrapper intentionally uses only the main classifier.
        result = model.load_state_dict(state, strict=False)
        unexpected = [key for key in result.unexpected_keys if not key.startswith("pair_classifier.")]
        if result.missing_keys or unexpected:
            fail("TinyGRU checkpoint does not match its model_config; missing="
                 f"{result.missing_keys}, unexpected={unexpected}")
        model.eval()
        return model, "tiny-gru-checkpoint"

    fail("PYTORCH_MODEL_LOADER_REQUIRED: checkpoint does not contain a supported "
         "model_config + model_state_dict layout. Provide --model-loader loader.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a supported PyTorch model to float32 TFLite")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-shape", help="For example: 1,90,3. Defaults to metadata input_shape.")
    parser.add_argument("--metadata", help="Optional preprocessing metadata JSON")
    parser.add_argument("--reference-input",
                        help="Optional .npy float32 tensor used for source/TFLite parity validation")
    parser.add_argument("--model-loader",
                        help="Trusted Python loader exposing load_model(path) or build_model()")
    parser.add_argument("--trust-pytorch-pickle", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import litert_torch
    except ImportError as error:
        fail("missing model-torch dependency: " + str(error))

    model_path = Path(args.model).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not model_path.is_file():
        fail(f"PyTorch model not found: {model_path}")
    metadata_path = find_metadata(model_path, args.metadata)
    input_shape = parse_shape(args.input_shape) if args.input_shape else shape_from_metadata(metadata_path)
    if input_shape is None:
        fail("input shape is unknown. Provide --input-shape or a metadata JSON containing input_shape.")

    loader_path = Path(args.model_loader).expanduser().resolve() if args.model_loader else None
    model, detected_format = load_model(torch, model_path, args.trust_pytorch_pickle, loader_path)
    if args.reference_input:
        reference_input_path = Path(args.reference_input).expanduser().resolve()
        if not reference_input_path.is_file():
            fail(f"reference input not found: {reference_input_path}")
        try:
            value = np.load(reference_input_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            fail(f"cannot load reference input .npy: {error}")
        if tuple(value.shape) != input_shape:
            fail("reference input shape does not match model contract: "
                 f"got {tuple(value.shape)}, expected {input_shape}")
        if not np.issubdtype(value.dtype, np.number):
            fail(f"reference input must be numeric, got {value.dtype}")
        sample = torch.from_numpy(np.asarray(value, dtype=np.float32))
        validation_input = str(reference_input_path)
    else:
        sample = torch.zeros(input_shape, dtype=torch.float32)
        validation_input = "synthetic_zeros"
    with torch.no_grad():
        reference = model(sample).detach().cpu().numpy()
        edge_model = litert_torch.convert(model, (sample,))
        converted = np.asarray(edge_model(sample))
    max_abs_error = float(np.max(np.abs(reference - converted)))
    if not np.allclose(reference, converted, rtol=1e-4, atol=1e-5):
        fail(f"PyTorch and LiteRT outputs differ: max_abs_error={max_abs_error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output))
    report = {
        "status": "success",
        "source_model": str(model_path),
        "source_format": detected_format,
        "tflite_model": str(output),
        "input_shape": list(input_shape),
        "metadata": str(metadata_path) if metadata_path else None,
        "model_loader": str(loader_path) if loader_path else None,
        "validation_input": validation_input,
        "max_abs_error": max_abs_error,
    }
    output.with_suffix(output.suffix + ".export.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
