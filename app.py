"""Streamlit UI for the local, single-PDF reading assistant."""

from html import escape
from pathlib import Path
import shutil
from uuid import uuid4

import streamlit as st

from src.agents.graph import build_graph
from src.agents.nodes import OllamaAgentModel
from src.config import Settings
from src.ingest import activate_pdf, active_index
from src.rag.retriever import open_retriever
from src.ui import answer_html, evidence_html, shorten_filename


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
active_pdf = settings.data_dir / "current.pdf"
sample_pdf = settings.data_dir / "sample_handbook.pdf"

st.set_page_config(page_title="Smart Document RAG", page_icon="✦", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#172033; --muted:#6f7b91; --line:#e7ebf2; --blue:#536dfe; --violet:#8b5cf6; --panel:#ffffff; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:linear-gradient(135deg,#f8faff 0%,#f4f5fb 52%,#faf8ff 100%); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1180px; padding:2.5rem 3rem 4rem; }
[data-testid="stSidebar"] { background:#111a2e; border-right:0; }
[data-testid="stSidebar"] * { color:#e8edfb; }
[data-testid="stSidebar"] .stCaption { color:#9eabc4; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:#1a2640; border:1px dashed #516286; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small { color:#aeb9d0; }
[data-testid="stSidebar"] .stButton > button { background:#243454; border:1px solid #3d5078; color:#f5f7ff; }
[data-testid="stSidebar"] .stButton > button:hover { border-color:#9aaeff; background:#304467; }
.brand { display:flex; align-items:center; gap:.8rem; margin-bottom:3rem; }
.brand-mark { display:grid; place-items:center; width:2.45rem; height:2.45rem; border-radius:12px; background:linear-gradient(135deg,#7388ff,#9b6cff); color:white; font-size:1.25rem; box-shadow:0 8px 24px #536dfe55; }
.brand-title { font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem; letter-spacing:-.02em; }
.eyebrow { color:#6677d9; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.7rem; }
.hero { animation:rise .5s ease both; padding:1rem 0 2rem; }
.hero h1 { font-family:'Space Grotesk'; font-size:clamp(2.2rem,5vw,4.2rem); line-height:1.02; letter-spacing:-.065em; margin:0; max-width:760px; }
.hero p { color:var(--muted); font-size:1.08rem; max-width:630px; margin:1.2rem 0 0; }
.status-card, .answer-card, .question-card { background:var(--panel); border:1px solid var(--line); border-radius:20px; box-shadow:0 14px 42px #26345e0b; }
.status-card { padding:1rem 1.05rem; margin:1.3rem 0; color:var(--ink); }
.status-label { color:#66738c; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }
.status-file { color:var(--ink); margin-top:.35rem; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
[data-testid="stSidebar"] .status-card { background:#fff; color:var(--ink); }
[data-testid="stSidebar"] .status-card .status-label { color:#66738c; }
[data-testid="stSidebar"] .status-card .status-file { color:var(--ink); }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#39c58a; margin-right:7px; box-shadow:0 0 0 4px #39c58a22; }
.question-card { padding:1.25rem 1.35rem .8rem; animation:rise .55s .05s ease both; }
div[data-testid="stForm"] { border:0; padding:0; }
.stTextInput input { color:var(--ink) !important; caret-color:var(--blue); border:1px solid #dce2ef; border-radius:12px; min-height:3rem; background:#fbfcff; font-size:1rem; }
.stTextInput input::placeholder { color:#8994a8 !important; opacity:1; }
.stTextInput input:focus { color:var(--ink) !important; border-color:var(--blue); box-shadow:0 0 0 3px #536dfe20; }
.stButton > button { border:0; border-radius:11px; min-height:2.75rem; font-weight:700; background:linear-gradient(135deg,#536dfe,#7656ee); color:#fff; box-shadow:0 8px 18px #536dfe30; transition:transform .18s, box-shadow .18s; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 12px 24px #536dfe44; }
.answer-card { padding:1.5rem 1.7rem; margin-top:1.6rem; animation:rise .4s ease both; }
.answer-heading { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1rem; }
.answer-heading h2 { font-family:'Space Grotesk'; margin:0; font-size:1.35rem; }
.route-pill { border-radius:999px; padding:.35rem .7rem; color:#5360a2; background:#eef0ff; font-size:.72rem; font-weight:700; white-space:nowrap; }
.answer-paragraph { color:#29344a !important; font-size:1.02rem; line-height:1.8; text-align:justify; margin:.8rem 0; }
.page-citation { display:inline-block; color:#5c54db; background:#eeecff; border:1px solid #dcd8ff; border-radius:5px; padding:.02rem .32rem; font-size:.78em; font-weight:700; white-space:nowrap; }
.evidence-card { background:#fbfcff; border:1px solid #e5e9f2; border-left:4px solid #8b7cf6; border-radius:13px; padding:1rem 1.05rem; margin:.75rem 0; animation:rise .35s ease both; }
.evidence-meta { display:flex; justify-content:space-between; color:#6868c8; font-size:.68rem; letter-spacing:.1em; font-weight:700; }
.evidence-text { color:#4d586d; line-height:1.65; margin-top:.55rem; font-size:.92rem; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:16px; background:#fff; margin-top:1rem; }
@keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@media (max-width:700px) { .block-container { padding:1.5rem 1rem 3rem; } .hero h1 { font-size:2.5rem; } .answer-heading { align-items:flex-start; flex-direction:column; } }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div class="brand-title">Smart Document<br>RAG</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="status-label">Document workspace</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")
    col1, col2 = st.columns(2)
    index_upload = col1.button("Index upload", disabled=upload is None, use_container_width=True)
    use_sample = col2.button("Use sample", disabled=not sample_pdf.exists(), use_container_width=True)

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
            with st.spinner("Preparing your document…"):
                count = activate_pdf(selected, active_pdf, settings, display_name=display_name)
            st.success(f"Ready · {count} chunks indexed")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

    if settings.manifest_path.exists():
        _, indexed_file = active_index(settings)
        st.markdown(
            f'<div class="status-card"><div class="status-label"><span class="status-dot"></span>Active document</div><div class="status-file">{escape(shorten_filename(indexed_file))}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("Your PDF and index stay on this device.")

st.markdown(
    '<section class="hero"><div class="eyebrow">Private document intelligence</div><h1>Ask your document<br>anything.</h1><p>Explore your PDF with clear, cited answers grounded in the pages you provide.</p></section>',
    unsafe_allow_html=True,
)

if not settings.manifest_path.exists():
    st.markdown(
        '<div class="status-card"><div class="status-label">Get started</div><div class="status-file">Upload a PDF from the sidebar to begin exploring.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown('<div class="question-card">', unsafe_allow_html=True)
with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder="What exception changes the usual approval deadline?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search document  →", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if submitted and question.strip():
    try:
        with st.spinner("Reading the relevant pages…"):
            graph = build_graph(open_retriever(settings), OllamaAgentModel(settings))
            result = graph.invoke({"question": question.strip()})
        route = "Refined search" if result["rewrite_count"] else "Direct search"
        st.markdown(
            f'<section class="answer-card"><div class="answer-heading"><h2>Your answer</h2><span class="route-pill">{route}</span></div>{answer_html(result["answer"])}</section>',
            unsafe_allow_html=True,
        )
        with st.expander("View retrieved evidence", expanded=False):
            st.caption("These are the source passages used to form the answer.")
            for chunk in result["chunks"]:
                st.markdown(evidence_html(chunk.page, chunk.text, chunk.score), unsafe_allow_html=True)
    except Exception as exc:
        st.error(
            "The local model or index could not be reached. Confirm Ollama is running "
            f"and both models are installed. Details: {exc}"
        )
