#!/usr/bin/env python3
"""Package the prototype source and notes into one shareable context archive."""

from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "context_package"
ARCHIVE_PATH = OUTPUT_DIR / "ai_runtime_prototype_context.zip"
TEXT_PATH = OUTPUT_DIR / "ai_runtime_prototype_context.md"

INCLUDED_SUFFIXES = {
    ".c",
    ".h",
    ".md",
    ".py",
    ".sh",
    ".mk",
    ".txt",
}
INCLUDED_NAMES = {"Makefile", ".gitignore"}
EXCLUDED_PARTS = {"build", ".git", "__pycache__", "context_package"}


def source_files() -> list[Path]:
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.name in INCLUDED_NAMES or path.suffix.lower() in INCLUDED_SUFFIXES:
            files.append(path)
    return sorted(files)


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def build_text(files: list[Path]) -> str:
    sections = [
        "# ai_runtime_prototype context",
        "",
        "Generated source and notes bundle. Binary files and build output are excluded.",
        "",
    ]
    for path in files:
        name = relative_path(path)
        language = {
            ".c": "c",
            ".h": "c",
            ".md": "markdown",
            ".py": "python",
            ".sh": "bash",
            ".mk": "makefile",
            ".txt": "text",
        }.get(path.suffix.lower(), "text")
        if path.name == "Makefile":
            language = "makefile"
        sections.extend(
            [
                f"## `{name}`",
                "",
                f"```{language}",
                path.read_text(encoding="utf-8", errors="replace").rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(sections)


def main() -> None:
    files = source_files()
    text = build_text(files)
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEXT_PATH.write_text(text, encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(TEXT_PATH, TEXT_PATH.name)
        for path in files:
            archive.write(path, relative_path(path))
    print(f"Packaged {len(files)} files")
    print(f"Text bundle: {TEXT_PATH}")
    print(f"Zip archive: {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()