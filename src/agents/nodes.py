"""The five small operations used by the bounded RAG graph."""

from __future__ import annotations

import re
from typing import Protocol, TypedDict

from llama_index.llms.ollama import Ollama

from src.config import Settings
from src.rag.retriever import RetrievedChunk
from src.tracing import trace_llm, trace_node


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


def _valid_statement_ids(statements: list[tuple[str, int]]) -> str:
    return ",".join(f"S{position}" for position in range(1, len(statements) + 1))


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
    # ponytail: this only rejects wholly unrelated grounded text; it is not a
    # semantic completeness proof.
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


class OllamaAgentModel:
    """Minimal prompt adapter around the local Ollama LLM."""

    def __init__(self, settings: Settings | None = None):
        settings = settings or Settings()
        self._llm = Ollama(
            model=settings.llm_model,
            base_url=settings.ollama_url,
            temperature=0.0,
            context_window=settings.ollama_context_window,
            request_timeout=120.0,
        )

    @trace_llm
    def _complete(self, prompt: str) -> str:
        return self._llm.complete(prompt).text.strip()

    def analyze(self, question: str) -> str:
        return self._complete(
            "Identify the core information need and useful PDF search terms. "
            "Be concise; do not answer the question.\n\nQuestion: " + question
        )

    def _correction_positions(
        self, question: str, statements: list[tuple[str, int]]
    ) -> list[int]:
        if re.match(r"\s*(?:compare|contrast)\b", question, re.IGNORECASE):
            return []
        evidence = _format_statement_evidence(statements)
        contradicts = self._complete(
            "Does the evidence explicitly contradict a factual assertion made in "
            "the question? A request to compare items makes no factual assertion. "
            "Reply only YES or NO.\n\n"
            f"Question: {question}\n\nEvidence statements:\n{evidence}"
        )
        if contradicts.strip().casefold() != "yes":
            return []
        correction = self._complete(
            "Which evidence statements directly correct a false claim in the "
            "question? Return only IDs from this valid list, comma-separated as "
            "needed, or exactly NONE: "
            f"{_valid_statement_ids(statements)}. Do not answer or explain.\n\n"
            f"Question: {question}\n\nEvidence statements:\n{evidence}"
        )
        return _selected_statement_positions(correction, len(statements))

    def _grade_once(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> bool:
        result = self._complete(
            "Decide whether the evidence contains enough information to answer the "
            "question without outside knowledge. Every requested part must be "
            "supported, including every subject in a comparison. Evidence that "
            "shows a premise in the question is false is sufficient to correct that "
            "premise; it does not need to explain why the false premise is true. "
            "For example, evidence that X is not Y is sufficient for a question "
            "asking why X is Y. "
            "Reply with exactly "
            "SUFFICIENT or INSUFFICIENT.\n\n"
            f"Question: {question}\n\nEvidence:\n"
            + format_evidence(chunks)
        )
        return _grade_is_sufficient(result)

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        if self._grade_once(question, chunks):
            return True

        # A holistic grade can miss direct support in a larger context. Focus the
        # model on individual statements and trust its explicit selection.
        statements = _rank_evidence_statements(
            question, _evidence_statements(chunks)
        )
        if self._correction_positions(question, statements):
            return True
        selection = self._complete(
            "Choose the smallest set of evidence statement IDs that collectively "
            "provides enough facts for a source-grounded answer to every part of the "
            "question. For comparisons, include support for every subject. A "
            "statement showing that a premise in the question is false answers that "
            "premise; select it instead of requiring evidence for the false claim. If a "
            "question asks why X is Y and a statement says X is not Y, select that "
            "statement. If any "
            "requested part is unsupported, return NONE. Return only comma-separated "
            "IDs from this valid list, or exactly NONE: "
            f"{_valid_statement_ids(statements)}. Do not answer or explain.\n\n"
            f"Question: {question}\n\nEvidence statements:\n"
            f"{_format_statement_evidence(statements)}"
        )
        selected = _selected_statement_positions(selection, len(statements))
        return bool(selected)

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
        prompt = (
            "Answer the question using only the retrieved PDF evidence. Write a "
            "natural, concise answer in one or more paragraphs. Explain the answer "
            "in your own words rather than copying or listing evidence sentences. "
            "Put a page citation such as [Page 2] at the end of every factual "
            "sentence. Do not invent facts or answer from outside knowledge. If the "
            "evidence is insufficient for any requested part, reply exactly: I don't "
            "have sufficient evidence in this PDF to answer that question.\n\n"
            f"Question: {question}\n\nRetrieved evidence:\n"
            + format_evidence(chunks)
        )
        draft = self._complete(prompt)
        if "[Page " in draft or draft.startswith("I don't have sufficient evidence"):
            return draft
        repaired = self._complete(
            "Add page citations to this answer using only the retrieved evidence. "
            "Keep the wording, do not add facts, and put at least one citation in "
            "each paragraph. Return only the revised answer. Use [Page N] exactly, "
            "not parentheses.\n\n"
            f"Question: {question}\n\nDraft answer:\n{draft}\n\n"
            f"Retrieved evidence:\n{format_evidence(chunks)}"
        )
        repaired = repaired.split("\n\nRetrieved evidence", 1)[0]
        return re.sub(r"\(Page (\d+)\)", r"[Page \1]", repaired)


class RAGNodes:
    def __init__(self, retriever: Retriever, model: AgentModel):
        self.retriever = retriever
        self.model = model

    @trace_node
    def analyze_question(self, state: AgentState) -> dict:
        return {
            "analysis": self.model.analyze(state["question"]),
            "query": state["question"],
            "rewrite_count": 0,
        }

    @trace_node
    def retrieve_chunks(self, state: AgentState) -> dict:
        return {"chunks": self.retriever.retrieve(state["query"])}

    @trace_node
    def grade_evidence(self, state: AgentState) -> dict:
        return {
            "evidence_good": self.model.grade(state["question"], state["chunks"])
        }

    @trace_node
    def rewrite_query(self, state: AgentState) -> dict:
        query = self.model.rewrite(
            state["question"], state["analysis"], state["chunks"]
        )
        return {"query": query, "rewrite_count": state["rewrite_count"] + 1}

    @trace_node
    def generate_answer(self, state: AgentState) -> dict:
        if not state["evidence_good"]:
            return {
                "answer": "I don't have sufficient evidence in this PDF to answer that question."
            }
        answer = self.model.answer(state["question"], state["chunks"])
        if not citations_are_grounded(answer, state["chunks"]) or not answer_mentions_question(
            answer, state["question"]
        ):
            answer = "I don't have sufficient evidence in this PDF to answer that question."
        return {"answer": answer}
