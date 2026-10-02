"""Pure answer and evidence formatting helpers for the RAG agent."""

from __future__ import annotations

import re

from src.rag.retriever import RetrievedChunk


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Page {chunk.page}]\n{chunk.text}" for chunk in chunks)


def _answer_claims(answer: str) -> list[str]:
    lines = answer.splitlines()
    content_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or re.fullmatch(r"(?:[-*_]\s*){3,}", stripped):
            continue
        if re.fullmatch(r"(?:=+|-+)", stripped):
            continue
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        if re.fullmatch(
            r"(?:\*\*|__)?(?:answer|response|summary):?(?:\*\*|__)?",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        stripped = re.sub(r"^(?:[-+*]\s+|\d+[.)]\s+)", "", stripped)
        content_lines.append(stripped)

    protected = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda match: match.group().replace(".", "\u0000"),
        "\n".join(content_lines),
    )
    protected = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|Fig|vs|etc)\.",
        lambda match: match.group().replace(".", "\u0000"),
        protected,
        flags=re.IGNORECASE,
    )
    return [
        claim.replace("\u0000", ".").strip()
        for claim in re.split(r"(?<=[.!?])(?:\s+|$)|\n+", protected)
        if claim.strip()
    ]


def _evidence_statements(
    chunks: list[RetrievedChunk],
) -> list[tuple[str, int]]:
    statements: list[tuple[str, int]] = []
    seen: set[str] = set()
    for chunk in chunks:
        for statement in _answer_claims(chunk.text.replace("\n", " ")):
            key = " ".join(statement.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            statements.append((statement, chunk.page))
    return statements


def _with_page_citation(statement: str, page: int) -> str:
    if statement.endswith((".", "!", "?")):
        return f"{statement[:-1]} [Page {page}]{statement[-1]}"
    return f"{statement} [Page {page}]"


def _format_statement_evidence(statements: list[tuple[str, int]]) -> str:
    return "\n".join(
        f"S{position}: [Page {page}] {statement}"
        for position, (statement, page) in enumerate(statements, start=1)
    )


def _valid_statement_ids(statements: list[tuple[str, int]]) -> str:
    return ",".join(f"S{position}" for position in range(1, len(statements) + 1))


def _search_terms(text: str) -> set[str]:
    stop_words = {
        "about", "already", "and", "but", "for", "from", "her", "his",
        "into", "she", "show", "story", "that", "the", "their", "this",
        "was", "were", "with",
    }
    terms = set()
    for word in re.findall(r"[a-z0-9]+", text.casefold()):
        if word.endswith("ied"):
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        if len(word) > 2 and word not in stop_words:
            terms.add(word)
    return terms


def _rank_evidence_statements(
    question: str, statements: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    question_terms = _search_terms(question)
    ranked = sorted(
        enumerate(statements),
        key=lambda item: (
            -len(question_terms & _search_terms(item[1][0])),
            item[0],
        ),
    )
    return [statement for _, statement in ranked]


def _selected_statement_positions(
    selection: str, statement_count: int, limit: int | None = None
) -> list[int]:
    selection = selection.strip()
    if selection.casefold() == "none":
        return []
    if not re.fullmatch(
        r"S\d+(?:\s*,\s*S\d+)*", selection, flags=re.IGNORECASE
    ):
        return []

    selected: list[int] = []
    for value in re.findall(r"S(\d+)", selection, flags=re.IGNORECASE):
        position = int(value)
        if not 1 <= position <= statement_count:
            return []
        if position not in selected:
            selected.append(position)
        if limit is not None and len(selected) == limit:
            break
    return selected


def answer_mentions_question(answer: str, question: str) -> bool:
    return bool(_search_terms(answer) & _search_terms(question))


def citations_are_grounded(answer: str, chunks: list[RetrievedChunk]) -> bool:
    evidence_by_page: dict[int, list[str]] = {}
    for chunk in chunks:
        evidence_by_page.setdefault(chunk.page, []).append(chunk.text)
    paragraphs = [paragraph.strip() for paragraph in answer.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return False

    for paragraph in paragraphs:
        cited_pages = {
            int(page) for page in re.findall(r"\[Page (\d+)\]", paragraph)
        }
        if not cited_pages or not cited_pages <= evidence_by_page.keys():
            return False
    return True
