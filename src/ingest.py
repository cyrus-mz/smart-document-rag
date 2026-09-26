"""Load one PDF, split it into page-aware chunks, and persist it in Chroma."""

from pathlib import Path
import json
import os
from uuid import uuid4

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import Settings


def active_index(settings: Settings) -> tuple[str, str]:
    """Return the active collection and display filename."""
    manifest = settings.manifest_path.read_text(encoding="utf-8")
    try:
        value = json.loads(manifest)
    except json.JSONDecodeError:
        return settings.collection_name, manifest.strip()
    return value["collection"], value["file_name"]


def _active_pdf(settings: Settings) -> Path | None:
    manifest = settings.manifest_path.read_text(encoding="utf-8")
    try:
        pdf_file = json.loads(manifest).get("pdf_file")
    except (json.JSONDecodeError, AttributeError):
        return None
    return settings.data_dir / pdf_file if pdf_file else None


def _activate_index(
    settings: Settings,
    collection: str,
    file_name: str,
    pdf_file: str,
) -> None:
    temporary_manifest = settings.manifest_path.with_name(
        f".{settings.manifest_path.name}.{uuid4().hex}.tmp"
    )
    try:
        temporary_manifest.write_text(
            json.dumps(
                {
                    "collection": collection,
                    "file_name": file_name,
                    "pdf_file": pdf_file,
                }
            ),
            encoding="utf-8",
        )
        os.replace(temporary_manifest, settings.manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)


def page_number(metadata: dict, fallback: int) -> int:
    """Return a stable, one-based PDF page number from reader metadata."""
    for key in ("source_page", "page_label", "page_number", "page"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            number = int(value)
            return number if number > 0 else number + 1
        except (TypeError, ValueError):
            continue
    return fallback


def load_and_chunk(pdf_path: Path, settings: Settings):
    """Use LlamaIndex for PDF loading and chunking while retaining page metadata."""
    documents = PDFReader().load_data(file=pdf_path)
    for position, document in enumerate(documents, start=1):
        document.metadata["source_page"] = page_number(document.metadata, position)
        document.metadata["file_name"] = pdf_path.name

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.get_nodes_from_documents(documents)


def ingest_pdf(
    pdf_path: Path,
    settings: Settings | None = None,
    display_name: str | None = None,
) -> int:
    """Replace the current local index with ``pdf_path`` and return chunk count."""
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    nodes = load_and_chunk(pdf_path, settings)

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    old_collection = (
        active_index(settings)[0] if settings.manifest_path.exists() else None
    )
    old_pdf = _active_pdf(settings) if settings.manifest_path.exists() else None
    new_collection = f"{settings.collection_name}__{uuid4().hex}"
    collection = client.create_collection(new_collection)
    try:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage = StorageContext.from_defaults(vector_store=vector_store)
        embedding = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.ollama_url,
            ollama_additional_kwargs={"num_ctx": settings.ollama_context_window},
        )
        VectorStoreIndex(nodes, storage_context=storage, embed_model=embedding)
        _activate_index(
            settings,
            new_collection,
            display_name or pdf_path.name,
            pdf_path.name,
        )
    except Exception:
        try:
            client.delete_collection(new_collection)
        except Exception:
            pass
        raise

    if old_collection and old_collection != new_collection:
        try:
            client.delete_collection(old_collection)
        except Exception:
            pass
    if old_pdf and old_pdf != pdf_path:
        try:
            old_pdf.unlink(missing_ok=True)
        except OSError:
            pass
    return len(nodes)


def activate_pdf(
    staged_pdf: Path,
    active_pdf: Path,
    settings: Settings | None = None,
    display_name: str | None = None,
) -> int:
    """Index a staged PDF and atomically make its resource pair active."""
    try:
        return ingest_pdf(
            staged_pdf,
            settings,
            display_name=display_name or active_pdf.name,
        )
    except Exception:
        staged_pdf.unlink(missing_ok=True)
        raise
