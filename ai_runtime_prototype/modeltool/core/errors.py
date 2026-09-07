class ModelToolError(RuntimeError):
    """A user-facing error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ModelToolError(code, message)
