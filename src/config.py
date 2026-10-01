"""Application settings, kept in one place for an easy local setup."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("SDA_DATA_DIR", "data"))
    collection_name: str = "document"
    llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_context_window: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_CONTEXT_WINDOW", "4096"))
    )
    chunk_size: int = 1024
    chunk_overlap: int = 100
    top_k: int = 4

    def __post_init__(self) -> None:
        if self.ollama_context_window <= 0:
            raise ValueError("OLLAMA_CONTEXT_WINDOW must be a positive integer")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "indexed_file.txt"
