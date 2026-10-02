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
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
dark = st.session_state.theme == "dark"
palette = {
    "ink": "#f3f6ff" if dark else "#172033",
    "muted": "#b8c2d8" if dark else "#526078",
    "line": "#33415c" if dark else "#dce2ef",
    "panel": "#17233a" if dark else "#ffffff",
    "app": "#0b1220" if dark else "#f8faff",
    "input": "#111d31" if dark else "#fbfcff",
    "secondary": "#c5cee0" if dark else "#4d586d",
    "accent": "#bfc8ff" if dark else "#4854b8",
    "placeholder": "#aeb9ce" if dark else "#66738c",
}
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{ --ink:{palette['ink']}; --muted:{palette['muted']}; --line:{palette['line']}; --panel:{palette['panel']}; --app:{palette['app']}; --input:{palette['input']}; --secondary:{palette['secondary']}; --blue:#8ea2ff; }}
html, body, [class*="css"] {{ font-family:'DM Sans', sans-serif; color:var(--ink); }}
[data-testid="stAppViewContainer"] {{ background:var(--app); }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1180px; padding:2.5rem 3rem 4rem; }}
[data-testid="stSidebar"] {{ background:#111a2e; border-right:0; }}
[data-testid="stSidebar"] * {{ color:#e8edfb; }}
[data-testid="stSidebar"] .stCaption {{ color:#d2daf0 !important; }}
[data-testid="stSidebar"] > div .status-label {{ color:#dbe3ff !important; }}
[data-testid="stExpander"] .stCaption {{ color:var(--secondary) !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background:#1a2640; border:1px dashed #7185b0; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {{ color:#d2daf0; }}
.brand {{ display:flex; align-items:center; gap:.8rem; margin-bottom:2rem; }}
.brand-mark {{ display:grid; place-items:center; width:2.45rem; height:2.45rem; border-radius:12px; background:linear-gradient(135deg,#7388ff,#9b6cff); color:white; font-size:1.25rem; }}
.brand-title {{ font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem; }}
.eyebrow, .status-label {{ color:{palette['accent']} !important; font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
.answer-heading h2 {{ color:var(--ink) !important; }}
[data-testid="stSidebar"] .status-card .status-label, [data-testid="stSidebar"] .status-card .status-file {{ color:var(--ink) !important; }}
.hero {{ animation:rise .5s ease both; padding:1rem 0 2rem; }}
.hero h1 {{ color:var(--ink) !important; font-family:'Space Grotesk'; font-size:clamp(2.2rem,5vw,4.2rem); line-height:1.02; letter-spacing:-.065em; margin:0; max-width:760px; }}
.hero p {{ color:var(--muted) !important; font-size:1.08rem; max-width:630px; margin:1.2rem 0 0; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{ color:#172033 !important; background:#f3f6ff !important; border-color:#aebce0 !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {{ color:#172033 !important; }}
.status-card, .answer-card, .question-card {{ background:var(--panel); border:1px solid var(--line); border-radius:20px; color:var(--ink); box-shadow:0 14px 42px #00000020; }}
.status-card {{ padding:1rem 1.05rem; margin:1.3rem 0; }}
.status-file {{ color:var(--ink); margin-top:.35rem; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.status-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:#52d6a0; margin-right:7px; }}
.question-card {{ padding:1.25rem 1.35rem .8rem; animation:rise .55s .05s ease both; }}
div[data-testid="stForm"] {{ border:0; padding:0; }}
.stTextArea textarea {{ color:var(--ink) !important; caret-color:#a8b7ff; border:1px solid var(--line); border-radius:12px; min-height:5.5rem; background:var(--input); font-size:1rem; line-height:1.5; overflow-wrap:anywhere; }}
.stTextArea textarea::placeholder {{ color:{palette['placeholder']} !important; opacity:1; }}
.stTextArea textarea:focus {{ color:var(--ink) !important; border-color:#9aabff; box-shadow:0 0 0 3px #8ea2ff44; }}
.stTextArea textarea {{ resize:vertical; }}
.stButton > button, button[kind="secondaryFormSubmit"] {{ border:1px solid #a9b6ff; border-radius:11px; min-height:2.75rem; font-weight:700; background:linear-gradient(135deg,#3f55d8,#6040c5) !important; color:#fff !important; box-shadow:0 8px 18px #536dfe40; transition:transform .18s, box-shadow .18s; }}
.stButton > button:hover, .stButton > button:focus, .stButton > button:active, button[kind="secondaryFormSubmit"]:hover, button[kind="secondaryFormSubmit"]:focus, button[kind="secondaryFormSubmit"]:active {{ color:#fff !important; background:linear-gradient(135deg,#3548c8,#5a3db5) !important; border-color:#d5dcff; }}
.stButton > button:hover *, .stButton > button:focus *, button[kind="secondaryFormSubmit"]:hover * {{ color:#fff !important; }}
.stButton > button:disabled {{ color:#aeb9ce !important; background:#35425b !important; border-color:#53627f; }}
[data-testid="stStatusWidget"], [data-testid="stSpinner"] {{ color:var(--ink) !important; }}
[data-testid="stSpinner"] svg, [data-testid="stSpinnerIcon"] {{ color:var(--ink) !important; stroke:var(--ink) !important; opacity:1 !important; }}
[data-testid="stSpinnerIcon"] {{ border-color:var(--ink) !important; border-right-color:transparent !important; }}
[data-testid="stSpinner"] * {{ color:var(--ink) !important; }}
.answer-card {{ padding:1.5rem 1.7rem; margin-top:1.6rem; animation:rise .4s ease both; }}
.answer-heading {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1rem; }}
.answer-heading h2 {{ font-family:'Space Grotesk'; margin:0; font-size:1.35rem; }}
.route-pill {{ border-radius:999px; padding:.35rem .7rem; color:#cbd4ff; background:#303b72; font-size:.72rem; font-weight:700; }}
.answer-paragraph {{ color:var(--ink) !important; font-size:1.02rem; line-height:1.8; text-align:justify; margin:.8rem 0; }}
.page-citation {{ display:inline-block; color:#d7dcff; background:#30356b; border:1px solid #626ee0; border-radius:5px; padding:.02rem .32rem; font-size:.78em; font-weight:700; }}
.evidence-card {{ background:var(--input); border:1px solid var(--line); border-left:4px solid #9a8cff; border-radius:13px; padding:1rem 1.05rem; margin:.75rem 0; animation:rise .35s ease both; }}
.evidence-meta {{ display:flex; justify-content:space-between; color:{palette['accent']} !important; font-size:.68rem; letter-spacing:.1em; font-weight:700; }}
.evidence-text {{ color:var(--secondary); line-height:1.65; margin-top:.55rem; font-size:.92rem; }}
[data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:16px; background:var(--panel); margin-top:1rem; color:var(--ink); }}
[data-testid="stExpander"] summary {{ color:var(--ink) !important; background:var(--panel) !important; border-radius:15px; }}
[data-testid="stExpander"] summary * {{ color:var(--ink) !important; background:transparent !important; }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation-duration:0.01ms !important; transition-duration:0.01ms !important; }} [data-testid="stSpinner"] svg, [data-testid="stSpinnerIcon"] {{ animation:none !important; opacity:1 !important; }} }}
@media (max-width:700px) {{ .block-container {{ padding:1.5rem 1rem 3rem; }} .hero h1 {{ font-size:2.5rem; }} .answer-heading {{ align-items:flex-start; flex-direction:column; }} }}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div class="brand-title">Smart Document<br>RAG</div></div>',
        unsafe_allow_html=True,
    )
    theme_label = "Switch to light theme" if dark else "Switch to dark theme"
    if st.button(theme_label, use_container_width=True, key="theme_toggle"):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()
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
            st.session_state.pop("answer_result", None)
            st.session_state.pop("question", None)
            st.session_state.pop("question_input", None)
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

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="What exception changes the usual approval deadline?",
        label_visibility="collapsed",
        height=96,
        help="Use Enter for a new line. Submit with the Search document button.",
        key="question_input",
    )
    submitted = st.form_submit_button("Search document  →", use_container_width=True)

if (submitted and question.strip()) or st.session_state.get("answer_result"):
    try:
        if submitted and question.strip():
            st.session_state.question = question
            with st.spinner("Reading the relevant pages…"):
                graph = build_graph(open_retriever(settings), OllamaAgentModel(settings))
                result = graph.invoke({"question": question.strip()})
            st.session_state.answer_result = result
        else:
            result = st.session_state.answer_result
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
