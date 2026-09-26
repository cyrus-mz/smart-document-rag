"""Small presentation helpers for the Streamlit UI."""

from pathlib import Path


def shorten_filename(file_name: str, max_length: int = 36) -> str:
    """Shorten a filename for display while retaining its final extension."""
    if len(file_name) <= max_length:
        return file_name

    suffix = Path(file_name).suffix
    available = max_length - len(suffix) - 1
    if available < 1:
        return file_name[: max(1, max_length - 1)] + "…"
    return file_name[:available] + "…" + suffix
