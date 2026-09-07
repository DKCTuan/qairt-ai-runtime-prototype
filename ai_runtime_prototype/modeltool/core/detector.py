from pathlib import Path

from .errors import ModelToolError


FORMAT_BY_SUFFIX = {
    ".tflite": "tflite",
    ".onnx": "onnx",
    ".pt": "pytorch",
    ".pth": "pytorch",
}


def detect_model_type(path: str | Path) -> str:
    source = Path(path).expanduser()
    try:
        return FORMAT_BY_SUFFIX[source.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(FORMAT_BY_SUFFIX))
        raise ModelToolError(
            "UNSUPPORTED_MODEL_FORMAT",
            f"unsupported model format {source.suffix or '<none>'}; supported: {supported}",
        ) from error
