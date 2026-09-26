"""Streamlit UI for the local, single-PDF reading assistant."""

from pathlib import Path
import shutil
from uuid import uuid4

import streamlit as st

from src.agents.graph import build_graph
from src.agents.nodes import OllamaAgentModel
from src.config import Settings
from src.ingest import activate_pdf, active_index
from src.rag.retriever import open_retriever
from src.ui import shorten_filename


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
active_pdf = settings.data_dir / "current.pdf"
sample_pdf = settings.data_dir / "sample_handbook.pdf"

st.set_page_config(page_title="Local PDF Reading Assistant", page_icon="📄")
st.title("📄 Local PDF Reading Assistant")
st.caption("One PDF · local Chroma index · bounded agent loop · Ollama")

with st.sidebar:
    st.header("Document")
    upload = st.file_uploader("Choose one PDF", type="pdf")
    col1, col2 = st.columns(2)
    index_upload = col1.button("Index upload", disabled=upload is None)
    use_sample = col2.button("Use sample", disabled=not sample_pdf.exists())

    try:
        selected: Path | None = None
        display_name: str | None = None
        if index_upload and upload is not None:
            selected = settings.data_dir / f".upload-{uuid4().hex}.pdf"
            selected.write_bytes(upload.getvalue())
            display_name = Path(upload.name).name
        elif use_sample:
            selected = settings.data_dir / f".sample-{uuid4().hex}.pdf"
            shutil.copyfile(sample_pdf, selected)
            display_name = sample_pdf.name

        if selected:
            with st.spinner("Reading, chunking, and embedding locally…"):
                count = activate_pdf(
                    selected,
                    active_pdf,
                    settings,
                    display_name=display_name,
                )
            st.success(f"Indexed {count} chunks")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

    if settings.manifest_path.exists():
        _, indexed_file = active_index(settings)
        st.text(
            f"Active index: {shorten_filename(indexed_file)}",
            help=f"Full filename: {indexed_file}",
        )
    st.caption(f"PDFs and vectors stay under `{settings.data_dir}`.")

if not settings.manifest_path.exists():
    st.warning("Index a PDF, or click **Use sample**, before asking a question.")
    st.stop()

with st.form("question_form"):
    question = st.text_input(
        "Ask about the document",
        placeholder="What exception changes the usual approval deadline?",
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Searching the PDF…"):
            graph = build_graph(open_retriever(settings), OllamaAgentModel(settings))
            result = graph.invoke({"question": question.strip()})
        st.subheader("Answer")
        st.markdown(result["answer"])
        route = "rewrote the query once" if result["rewrite_count"] else "direct retrieval"
        st.caption(f"Route: {route}")
        with st.expander("Retrieved evidence"):
            for chunk in result["chunks"]:
                score = "" if chunk.score is None else f" · score {chunk.score:.3f}"
                st.markdown(f"**Page {chunk.page}{score}**")
                st.write(chunk.text)
    except Exception as exc:
        st.error(
            "The local model or index could not be reached. Confirm Ollama is running "
            f"and both models are installed. Details: {exc}"
        )
