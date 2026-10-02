"""Thin LlamaIndex/Chroma retrieval boundary."""

from dataclasses import dataclass
from typing import Any

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import Settings
from src.ingest import active_index
from src.tracing import trace_retrieval


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page: int
    score: float | None = None


def to_chunk(result: Any) -> RetrievedChunk:
    """Convert a LlamaIndex NodeWithScore-like result into app data."""
    metadata = result.node.metadata
    return RetrievedChunk(
        text=result.node.get_content(),
        page=int(metadata["source_page"]),
        score=result.score,
    )


class LlamaIndexRetriever:
    def __init__(self, index: Any, top_k: int = 4):
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    @trace_retrieval
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [to_chunk(result) for result in self._retriever.retrieve(query)]


def open_retriever(settings: Settings | None = None) -> LlamaIndexRetriever:
    settings = settings or Settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection_name, _ = active_index(settings)
    collection = client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embedding = OllamaEmbedding(
        model_name=settings.embedding_model,
        base_url=settings.ollama_url,
        ollama_additional_kwargs={"num_ctx": settings.ollama_context_window},
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding)
    return LlamaIndexRetriever(index, settings.top_k)
