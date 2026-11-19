"""Resume -> Job Matcher (Streamlit). Run locally: streamlit run streamlit_app.py

Everything runs on CPU without PyTorch (ONNX models), so it fits Streamlit Community Cloud.
The resume is processed in memory and never stored; telemetry keeps numbers only.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jobmatch.config import (
    ALPHA,
    EXPLAIN_TOP,
    INDEX_REPO,
    INDEX_VERSION,
    MODE,
    SAMPLE_RESUME,
    SKILL_MODEL,
    TAU,
)
from jobmatch.index import clean_skills
from jobmatch.telemetry import Telemetry
from jobmatch.ui import render_footer, render_match

TELEMETRY = Path(tempfile.gettempdir()) / "jobmatch_telemetry.jsonl"

st.set_page_config(page_title="Resume → Job Matcher", page_icon="🧭", layout="wide")


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file locally
        from jobmatch.llm import load_api_key

        return load_api_key(name)


@st.cache_resource(show_spinner="Loading models and the job index (first visit only, ~1 min)...")
def load_matcher():
    from huggingface_hub import snapshot_download
    from skillex.onnx_predict import OnnxSkillExtractor

    from jobmatch.embed import OnnxEmbedder
    from jobmatch.index import JobIndex
    from jobmatch.llm import FallbackLLM, groq, llama_server
    from jobmatch.pipeline import Matcher

    root = snapshot_download(INDEX_REPO, repo_type="dataset", allow_patterns=[f"{INDEX_VERSION}/*"])
    index = JobIndex.load(f"{root}/{INDEX_VERSION}", mmap=True)
    emb = OnnxEmbedder(threads=2)
    ext = OnnxSkillExtractor(SKILL_MODEL, threads=2, low_memory=True)
    key = _secret("GROQ_API_KEY")
    # One quick retry on a rate limit (free tier is per minute); otherwise fail fast and show the fallback.
    llm = groq(key, timeout=25, rate_limit_retries=1, max_wait=3.0) if key else None
    # Optional: the distilled student (explain-distill, a llama-server on CPU) as the unlimited fallback. Groq stays
    # first: it answers in ~2 s, the 0.6B student needs ~25-60 s on CPU (explain-distill serving benchmark).
    if student_url := _secret("STUDENT_URL"):
        student = llama_server(student_url, api_key=_secret("STUDENT_API_KEY"), timeout=120)
        llm = FallbackLLM(llm, student) if llm else student
    return Matcher(
        index,
        embed_query=emb.query,
        embed_phrases=emb.encode,
        extract_skills=lambda text: [s["text"] for s in clean_skills(ext.extract(text)["spans"])],
        llm=llm,
        telemetry=Telemetry(TELEMETRY),
        alpha=ALPHA,
        tau=TAU,
    )


st.title("🧭 Resume → Job Matcher")
st.caption(
    "Paste a resume and get the best-matching tech jobs from 7,817 LinkedIn postings (2023–24), with the skills you "
    "have, the ones you're missing, and quotes from each posting as evidence. "
    "**Your resume is processed in memory and never stored.**"
)

with st.sidebar:
    st.header("Settings")
    k = st.slider("Jobs to show", 3, 10, 5)
    modes = ["bm25", "hybrid", "dense"]
    mode = st.radio("Retrieval", modes, index=modes.index(MODE),
                    help="bm25 = keywords · dense = embeddings · hybrid = both, fused with RRF. Default chosen on the dev set.")
    skill_rerank = st.toggle("Skill-coverage re-rank", value=True)
    explain_on = st.toggle("LLM explanations", value=True, help=f"For the top {EXPLAIN_TOP} matches (free-tier LLM limits)")
    st.divider()
    st.markdown(
        "**How it works:** BM25 + bge-small embeddings → Reciprocal Rank Fusion → re-rank by how many of the job's "
        "skills your resume covers (skills from a fine-tuned DeBERTa-v3 + CRF) → an LLM explains the top matches, "
        "and every quote is checked against the posting.\n\n"
        "[Code](https://github.com/MSR327/resume-job-matcher) · "
        "[Skill model](https://huggingface.co/srish327/deberta-v3-skillspan-crf) · "
        "[Job index](https://huggingface.co/datasets/srish327/tech-jobs-index)"
    )

if "resume" not in st.session_state:
    st.session_state.resume = ""
if st.button("Use a sample resume"):
    st.session_state.resume = SAMPLE_RESUME
resume = st.text_area("Resume (plain text)", key="resume", height=240,
                      placeholder="Paste your resume here: experience, projects, skills...")
go = st.button("Find matching jobs", type="primary", disabled=len(resume.split()) < 20)
if len(resume.split()) < 20:
    st.caption("Paste at least ~20 words.")

if go:
    matcher = load_matcher()
    if explain_on and matcher.llm is None:
        st.warning("No LLM key configured, so showing ranked jobs and skill overlap without explanations.")
    t0 = time.perf_counter()
    with st.spinner("Matching..."):
        matches = matcher.match(resume, k=k, mode=mode, skill_rerank=skill_rerank, explain_top=explain_on,
                                explain_n=EXPLAIN_TOP)
    elapsed = time.perf_counter() - t0

    for rank, m in enumerate(matches, 1):
        render_match(rank, m)
    render_footer(matches, elapsed, INDEX_VERSION)
