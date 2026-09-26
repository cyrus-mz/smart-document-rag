"""The five small operations used by the bounded RAG graph."""

from __future__ import annotations

import re
from typing import Protocol, TypedDict

from llama_index.llms.ollama import Ollama

from src.config import Settings
from src.rag.retriever import RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str
    query: str
    analysis: str
    chunks: list[RetrievedChunk]
    evidence_good: bool
    rewrite_count: int
    answer: str


class Retriever(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class AgentModel(Protocol):
    def analyze(self, question: str) -> str: ...
    def grade(self, question: str, chunks: list[RetrievedChunk]) -> bool: ...
    def rewrite(
        self, question: str, analysis: str, chunks: list[RetrievedChunk]
    ) -> str: ...
    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str: ...


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Page {chunk.page}]\n{chunk.text}" for chunk in chunks)


def _grade_is_sufficient(result: str) -> bool:
    first_line = next((line for line in result.splitlines() if line.strip()), "")
    words = re.findall(r"[A-Z]+", first_line.upper())
    return words == ["SUFFICIENT"]


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


def _search_terms(text: str) -> set[str]:
    stop_words = {
        "about",
        "already",
        "and",
        "but",
        "for",
        "from",
        "her",
        "his",
        "into",
        "she",
        "show",
        "story",
        "that",
        "the",
        "their",
        "this",
        "was",
        "were",
        "with",
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
    selection: str, statement_count: int, limit: int = 3
) -> list[int]:
    selected: list[int] = []
    for value in re.findall(r"\bS(\d+)\b", selection, flags=re.IGNORECASE):
        position = int(value)
        if 1 <= position <= statement_count and position not in selected:
            selected.append(position)
        if len(selected) == limit:
            break
    return selected


def citations_are_grounded(answer: str, chunks: list[RetrievedChunk]) -> bool:
    evidence_by_page: dict[int, list[str]] = {}
    for chunk in chunks:
        evidence_by_page.setdefault(chunk.page, []).append(chunk.text)
    claims = _answer_claims(answer)
    if not claims:
        return False

    for claim in claims:
        cited_pages = {
            int(page) for page in re.findall(r"\[Page (\d+)\]", claim)
        }
        if not cited_pages or not cited_pages <= evidence_by_page.keys():
            return False
        claim_text = re.sub(r"\[Page \d+\]", "", claim)
        normalized_claim = " ".join(
            re.sub(r"[^\w\s]", " ", claim_text.lower()).split()
        )
        cited_sentences = [
            " ".join(re.sub(r"[^\w\s]", " ", sentence.lower()).split())
            for page in cited_pages
            for text in evidence_by_page[page]
            for sentence in _answer_claims(text.replace("\n", " "))
        ]
        if not normalized_claim or not any(
            normalized_claim == sentence for sentence in cited_sentences
        ):
            return False
    return True


class OllamaAgentModel:
    """Minimal prompt adapter around the local Ollama LLM."""

    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        self._llm = Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            context_window=settings.ollama_context_window,
            request_timeout=120.0,
        )

    def _complete(self, prompt: str) -> str:
        return self._llm.complete(prompt).text.strip()

    def analyze(self, question: str) -> str:
        return self._complete(
            "Identify the core information need and useful PDF search terms. "
            "Be concise; do not answer the question.\n\nQuestion: " + question
        )

    def _grade_once(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> bool:
        result = self._complete(
            "Decide whether the evidence contains enough information to answer the "
            "question without outside knowledge. Reply with exactly SUFFICIENT or "
            f"INSUFFICIENT.\n\nQuestion: {question}\n\nEvidence:\n"
            + format_evidence(chunks)
        )
        return _grade_is_sufficient(result)

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        if self._grade_once(question, chunks):
            return True

        # A holistic grade can miss direct support in a larger context. Focus it
        # on selected statements, then apply the same sufficiency decision again.
        statements = _rank_evidence_statements(
            question, _evidence_statements(chunks)
        )
        selection = self._complete(
            "Choose up to three evidence statement IDs that collectively provide "
            "enough facts for a useful, source-grounded answer to the question. A "
            "statement that corrects a mistaken premise counts as support. Return "
            "only comma-separated IDs such as S2,S5, or NONE when the requested "
            "information is absent. Do not answer or explain.\n\n"
            f"Question: {question}\n\nEvidence statements:\n"
            f"{_format_statement_evidence(statements)}"
        )
        selected = _selected_statement_positions(selection, len(statements))
        if not selected:
            return False
        focused_evidence = [
            RetrievedChunk(statements[position - 1][0], statements[position - 1][1])
            for position in selected
        ]
        return self._grade_once(question, focused_evidence)

    def rewrite(
        self, question: str, analysis: str, chunks: list[RetrievedChunk]
    ) -> str:
        return self._complete(
            "Rewrite the question as one improved semantic-search query. Return only "
            f"the query.\n\nQuestion: {question}\nIntent: {analysis}\n\n"
            "Previously weak evidence:\n"
            + format_evidence(chunks)
        )

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        statements = _rank_evidence_statements(
            question, _evidence_statements(chunks)
        )
        if not statements:
            return ""
        selection = self._complete(
            "Select up to three evidence statement IDs that directly answer the "
            "question. Rank the IDs from most to least useful, covering the initial "
            "situation, central event, and outcome when available. Ignore tangential "
            "passages. Return only comma-separated IDs such as S2,S5, or NONE when "
            "the evidence cannot answer. Do not answer, paraphrase, or explain.\n\n"
            f"Question: {question}\n\nEvidence statements:\n"
            f"{_format_statement_evidence(statements)}"
        )
        selected = _selected_statement_positions(selection, len(statements))
        return "\n".join(
            _with_page_citation(*statements[position - 1])
            for position in selected
        )


class RAGNodes:
    def __init__(self, retriever: Retriever, model: AgentModel):
        self.retriever = retriever
        self.model = model

    def analyze_question(self, state: AgentState) -> dict:
        return {
            "analysis": self.model.analyze(state["question"]),
            "query": state["question"],
            "rewrite_count": 0,
        }

    def retrieve_chunks(self, state: AgentState) -> dict:
        return {"chunks": self.retriever.retrieve(state["query"])}

    def grade_evidence(self, state: AgentState) -> dict:
        return {
            "evidence_good": self.model.grade(state["question"], state["chunks"])
        }

    def rewrite_query(self, state: AgentState) -> dict:
        query = self.model.rewrite(
            state["question"], state["analysis"], state["chunks"]
        )
        return {"query": query, "rewrite_count": state["rewrite_count"] + 1}

    def generate_answer(self, state: AgentState) -> dict:
        if not state["evidence_good"]:
            return {
                "answer": "I don't have sufficient evidence in this PDF to answer that question."
            }
        answer = self.model.answer(state["question"], state["chunks"])
        if not citations_are_grounded(answer, state["chunks"]):
            answer = (
                "I couldn't produce a source-grounded answer from the retrieved "
                "evidence. Try rephrasing the question."
            )
        return {"answer": answer}
