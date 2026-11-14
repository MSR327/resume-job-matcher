import json
import zlib

import numpy as np
import pandas as pd

from jobmatch.index import build_index
from jobmatch.llm import FakeLLM, LLMError
from jobmatch.pipeline import Matcher
from jobmatch.telemetry import Telemetry

DIM = 32


def _embed(texts):
    """Same text (case-insensitive) -> same unit vector; different texts -> near-orthogonal."""
    v = np.stack([np.random.default_rng(zlib.crc32(t.lower().encode())).normal(size=DIM) for t in texts])
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _skills_of(text):
    return [w.strip(".,") for w in text.split() if w[:1].isupper()]


JOBS = pd.DataFrame({
    "job_id": ["a", "b", "c"],
    "title": ["backend engineer", "frontend developer", "devops engineer"],
    "company": ["X", "Y", "Z"], "location": ["", "", ""],
    "description": ["build Python and Django apis.", "build React and TypeScript uis.",
                    "run Kubernetes and Terraform on AWS."],
    "url": ["", "", ""], "listed_at": [0, 0, 0],
})


def _matcher(llm=None, telemetry=None):
    idx = build_index(JOBS, _embed, lambda ds: [[{"type": "KNOWLEDGE", "text": s} for s in _skills_of(d)] for d in ds],
                      version="t")
    return Matcher(idx, embed_query=lambda q: _embed([q])[0], embed_phrases=_embed, extract_skills=_skills_of,
                   llm=llm, telemetry=telemetry)


def test_match_ranks_and_reports_skill_overlap():
    out = _matcher().match("I know Kubernetes and Terraform, some Python", k=2, mode="bm25")
    top = out[0]
    assert top.job["job_id"] == "c"
    assert top.matched == ["Kubernetes", "Terraform"] and top.missing == ["AWS"]
    assert abs(top.coverage - 2 / 3) < 1e-6 and top.explanation is None  # no LLM configured


def test_match_explains_top_k_and_logs_numbers(tmp_path):
    reply = json.dumps({"fit_summary": "ok", "matched_skills": ["Kubernetes"], "missing_skills": ["AWS"],
                        "evidence": [{"claim": "k8s", "quote": "run Kubernetes and Terraform"}]})
    tel = Telemetry(tmp_path / "t.jsonl")
    def by_job(messages):  # devops job gets a good answer, the other job's LLM call fails
        return reply if "devops engineer" in messages[-1]["content"] else LLMError("429")

    out = _matcher(FakeLLM(by_job), tel).match("I know Kubernetes and Terraform, some Python", k=2, mode="bm25")
    assert out[0].job["job_id"] == "c"
    assert out[0].explanation.ok and out[0].explanation.citations_kept == 1
    assert not out[1].explanation.ok  # fell back, still has skill overlap
    row = tel.read()[0]
    assert row["llm_ok"] == 1 and row["llm_failed"] == 1 and row["citations_kept"] == 1
    assert all(isinstance(v, (int, float)) for v in row.values())


def test_skill_rerank_can_be_disabled():
    m = _matcher()
    a = [x.job["job_id"] for x in m.match("Python Django", k=3, mode="bm25", skill_rerank=False)]
    assert a[0] == "a"


def test_explain_only_first_n():
    reply = json.dumps({"fit_summary": "ok", "matched_skills": [], "missing_skills": [], "evidence": []})
    fake = FakeLLM(lambda messages: reply)
    out = _matcher(fake).match("I know Kubernetes and Terraform, some Python", k=2, mode="bm25", explain_n=1)
    assert out[0].explanation.ok and out[1].explanation is None and len(fake.calls) == 1
