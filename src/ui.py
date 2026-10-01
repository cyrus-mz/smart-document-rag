"""Small presentation helpers for the Streamlit UI."""

from html import escape
from pathlib import Path
import re


def shorten_filename(file_name: str, max_length: int = 36) -> str:
    """Shorten a filename for display while retaining its final extension."""
    if len(file_name) <= max_length:
        return file_name

    suffix = Path(file_name).suffix
    available = max_length - len(suffix) - 1
    if available < 1:
        return file_name[: max(1, max_length - 1)] + "…"
    return file_name[:available] + "…" + suffix


def answer_html(answer: str) -> str:
    """Render model text with safe, highlighted page citations."""
    text = escape(answer)
    text = re.sub(
        r"\[Page (\d+)\]",
        r'<span class="page-citation">Page \1</span>',
        text,
    )
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return "".join(f'<div class="answer-paragraph">{part}</div>' for part in paragraphs)


def evidence_html(page: int, text: str, score: float | None) -> str:
    """Render one retrieved chunk as a compact evidence card."""
    score_text = "" if score is None else f"{score:.3f} match"
    return (
        '<article class="evidence-card">'
        f'<div class="evidence-meta"><span>PAGE {page}</span>'
        f'<span>{escape(score_text)}</span></div>'
        f'<div class="evidence-text">{escape(text)}</div>'
        "</article>"
    )
