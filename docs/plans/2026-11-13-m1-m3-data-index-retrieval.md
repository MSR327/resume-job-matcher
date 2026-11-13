# Resume → Job Matcher, Milestones 1–3: Data, Index, Retrieval Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans (inline) to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the LinkedIn postings dataset into a versioned, searchable tech-job index. Ship hybrid retrieval (BM25 + dense + RRF) and skill-coverage re-ranking as tested library code.

**Architecture:** Pure-Python library `jobmatch` (src layout). Heavy steps (embedding and skill extraction over ~15k jobs) run in Kaggle notebooks that call the same library. The artifacts go to the HF dataset repo `srish327/tech-jobs-index`. Everything else is unit-tested locally with tiny fixtures, with no network and no models.

**Tech Stack:** Python 3.12, pandas + pyarrow, bm25s, sentence-transformers (`BAAI/bge-small-en-v1.5`), numpy, `skillex` (from github.com/MSR327/skillspan-ner), pytest, ruff.

**Spec:** `docs/specs/2026-11-13-resume-job-matcher-design.md`

## Global Constraints
- $0 compute/hosting. GPU work runs on Kaggle; launch it through a Sonnet subagent driving Chrome.
- Data: arshkon LinkedIn Job Postings 2023–24, CC BY-SA 4.0. The derived index is published CC BY-SA 4.0 with attribution.
- Never store or log resume text. Telemetry holds numbers only.
- Commits are authored by the user. **No `Co-Authored-By: Claude` or "Generated with Claude Code" lines.**
- Unit tests: no network, no model downloads (CI runs them on CPU in under a minute).
- Local venv: `V=/private/tmp/claude-501/-Users-msr-claude-code-hug/274362c5-a0fe-4a6f-a4b0-8ffd3c7687d6/scratchpad/venv/bin`

## File map
| File | Responsibility |
|---|---|
| `pyproject.toml` | package + deps (`[index]` extra for heavy deps) |
| `src/jobmatch/corpus.py` | tech-title filter, text cleaning, dedupe → corpus DataFrame |
| `src/jobmatch/index.py` | build / save / load `JobIndex` (BM25, vectors, skills, manifest) |
| `src/jobmatch/retrieve.py` | `rrf()`, `HybridRetriever.search()` |
| `src/jobmatch/rerank.py` | `skill_coverage()`, `rerank()` |
| `notebooks/01_build_index.ipynb` | Kaggle: corpus → index → push to HF |
| `DATA_CARD.md` | dataset card for `srish327/tech-jobs-index` |
| `tests/test_corpus.py`, `tests/test_retrieve.py`, `tests/test_rerank.py`, `tests/test_index.py` | unit tests |
| `.github/workflows/ci.yml` | ruff + pytest |

---

### Task 1: Scaffold + corpus filtering

**Files:** Create `pyproject.toml`, `src/jobmatch/__init__.py`, `src/jobmatch/corpus.py`, `tests/test_corpus.py`, `.github/workflows/ci.yml`, `.gitignore`, `LICENSE` (MIT).

**Interfaces — Produces:**
- `is_tech_title(title: str) -> bool`
- `clean_text(text: str) -> str`
- `build_corpus(df: pd.DataFrame, min_desc_chars: int = 400) -> pd.DataFrame` with columns
  `job_id:str, title, company, location, description, url, listed_at:int`, deduped, `job_id` unique.

- [ ] **Step 1: Write failing tests** (`tests/test_corpus.py`)
```python
import pandas as pd
from jobmatch.corpus import build_corpus, clean_text, is_tech_title

def test_tech_titles():
    for t in ["Senior Software Engineer", "Data Scientist II", "ML Engineer", "DevOps Engineer",
              "Frontend Developer (React)", "Site Reliability Engineer", "QA Automation Engineer"]:
        assert is_tech_title(t), t
    for t in ["Sales Engineer", "Registered Nurse", "Mechanical Engineer", "Technical Recruiter",
              "Civil Engineer", "Account Executive"]:
        assert not is_tech_title(t), t

def test_clean_text_collapses_whitespace_and_bullets():
    assert clean_text("Hello  world\n\n\n• Python\r\n") == "Hello world\n\n- Python"

def _row(i, title, desc, company="Acme"):
    return {"job_id": i, "title": title, "company_name": company, "location": "Remote",
            "description": desc, "job_posting_url": f"https://x/{i}", "listed_time": 1700000000000}

def test_build_corpus_filters_and_dedupes():
    long = "Build APIs in Python and Go. " * 30
    df = pd.DataFrame([
        _row(1, "Software Engineer", long),
        _row(2, "Software Engineer", long),            # duplicate (same company+title+desc)
        _row(3, "Software Engineer", "too short"),     # short description
        _row(4, "Registered Nurse", long),             # not tech
        _row(5, "Data Engineer", long, company="Beta"),
    ])
    out = build_corpus(df)
    assert out["job_id"].tolist() == ["1", "5"]
    assert set(out.columns) == {"job_id", "title", "company", "location", "description", "url", "listed_at"}
```
- [ ] **Step 2:** Run `PYTHONPATH=src $V/python -m pytest tests/test_corpus.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `src/jobmatch/corpus.py`**
```python
"""Filter the LinkedIn postings dump to clean, deduplicated tech jobs."""
from __future__ import annotations
import hashlib
import re
import pandas as pd

_INCLUDE = re.compile(
    r"\b(software|developer|programmer|full[- ]?stack|front[- ]?end|back[- ]?end|web|mobile|ios|android|"
    r"data (scientist|engineer|analyst|architect)|machine learning|\bml\b|\bai\b|deep learning|nlp|"
    r"devops|site reliability|\bsre\b|cloud|platform engineer|infrastructure engineer|"
    r"security engineer|cyber ?security|application security|qa|quality assurance|test automation|sdet|"
    r"database|\bdba\b|analytics engineer|solutions architect|software architect|computer vision)\b",
    re.I,
)
_EXCLUDE = re.compile(r"\b(sales|recruit|account (executive|manager)|nurse|mechanical|civil|electrical|"
                      r"chemical|manufacturing|marketing|teacher|driver)\b", re.I)

def is_tech_title(title: str) -> bool:
    return bool(_INCLUDE.search(title or "")) and not _EXCLUDE.search(title or "")

def clean_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    t = re.sub(r"^[ \t]*[•●▪◦*·][ \t]*", "- ", t, flags=re.M)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def build_corpus(df: pd.DataFrame, min_desc_chars: int = 400) -> pd.DataFrame:
    out = pd.DataFrame({
        "job_id": df["job_id"].astype(str),
        "title": df["title"].fillna("").map(str.strip),
        "company": df["company_name"].fillna("").map(str.strip),
        "location": df["location"].fillna(""),
        "description": df["description"].fillna("").map(clean_text),
        "url": df["job_posting_url"].fillna(""),
        "listed_at": df["listed_time"].fillna(0).astype("int64"),
    })
    out = out[out["title"].map(is_tech_title) & (out["description"].str.len() >= min_desc_chars)]
    key = (out["company"].str.lower() + "|" + out["title"].str.lower() + "|"
           + out["description"].map(lambda d: hashlib.md5(d.lower().encode()).hexdigest()))
    out = out.loc[~key.duplicated()].drop_duplicates("job_id")
    return out.reset_index(drop=True)
```
- [ ] **Step 4:** Run the tests. Expected: PASS. If `test_tech_titles` fails, adjust the regexes (not the tests).
- [ ] **Step 5: Scaffold.** `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "jobmatch"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy", "pandas", "pyarrow", "bm25s>=0.2"]
[project.optional-dependencies]
index = ["sentence-transformers>=3", "skillex @ git+https://github.com/MSR327/skillspan-ner.git", "huggingface_hub"]
dev = ["pytest", "ruff"]
[tool.setuptools.packages.find]
where = ["src"]
[tool.pytest.ini_options]
pythonpath = ["src"]
[tool.ruff]
line-length = 110
```
`.github/workflows/ci.yml`: checkout, setup-python 3.12, `pip install -e ".[dev]"`, `ruff check .`, `pytest -q`.
`.gitignore`: `__pycache__/`, `.venv/`, `*.egg-info/`, `artifacts/`, `.streamlit/secrets.toml`.
- [ ] **Step 6:** `$V/ruff check . && PYTHONPATH=src $V/python -m pytest -q` → all pass. Commit `"Scaffold jobmatch; tech-job corpus filter with tests"`, then push.

---

### Task 2: Hybrid retrieval (BM25 + dense + RRF)

**Files:** Create `src/jobmatch/retrieve.py`, `tests/test_retrieve.py`.

**Interfaces — Produces:**
- `rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]`: doc index and fused score, best first.
- `class HybridRetriever(bm25, vectors: np.ndarray, embed_query: Callable[[str], np.ndarray])` with
  `.search(text: str, k: int = 50, depth: int = 100, mode: str = "hybrid") -> list[tuple[int, float]]`,
  where `mode` is one of `bm25 | dense | hybrid`. `bm25` is a `bm25s.BM25`.

- [ ] **Step 1: Failing tests**
```python
import bm25s
import numpy as np
from jobmatch.retrieve import HybridRetriever, rrf

def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert [d for d, _ in fused[:2]] in ([1, 2], [2, 1])
    assert dict(fused)[3] < dict(fused)[1]
    assert {d for d, _ in fused} == {1, 2, 3, 4}

DOCS = ["python django rest api backend", "react typescript frontend ui", "kubernetes terraform aws devops"]

def _retriever():
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(DOCS, stopwords="en", show_progress=False), show_progress=False)
    vecs = np.eye(3, dtype=np.float32)               # doc i <-> axis i
    def embed(q):                                     # toy "semantic" embedder
        return np.array([("backend" in q) + 0.0, ("ui" in q) + 0.0, ("cloud" in q) + 0.0], np.float32)
    return HybridRetriever(bm, vecs, embed)

def test_modes():
    r = _retriever()
    assert r.search("terraform", k=1, mode="bm25")[0][0] == 2
    assert r.search("cloud infra", k=1, mode="dense")[0][0] == 2      # no word overlap; dense finds it
    assert r.search("python cloud", k=2, mode="hybrid")[0][0] in (0, 2)
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**
```python
"""Hybrid retrieval: BM25 and dense rankings fused with Reciprocal Rank Fusion."""
from __future__ import annotations
from collections.abc import Callable
import bm25s
import numpy as np

def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

class HybridRetriever:
    def __init__(self, bm25: bm25s.BM25, vectors: np.ndarray, embed_query: Callable[[str], np.ndarray]):
        self.bm25, self.vectors, self.embed_query = bm25, vectors, embed_query

    def _bm25(self, text: str, depth: int) -> list[int]:
        depth = min(depth, self.vectors.shape[0])
        docs, scores = self.bm25.retrieve(bm25s.tokenize([text], stopwords="en", show_progress=False),
                                          k=depth, show_progress=False)
        return [int(d) for d, s in zip(docs[0], scores[0]) if s > 0]

    def _dense(self, text: str, depth: int) -> list[int]:
        sims = self.vectors @ self.embed_query(text)
        return np.argsort(-sims)[:depth].tolist()

    def search(self, text: str, k: int = 50, depth: int = 100, mode: str = "hybrid") -> list[tuple[int, float]]:
        if mode == "bm25":
            return [(d, 1.0 / (r + 1)) for r, d in enumerate(self._bm25(text, depth))][:k]
        if mode == "dense":
            return [(d, 1.0 / (r + 1)) for r, d in enumerate(self._dense(text, depth))][:k]
        if mode == "hybrid":
            return rrf([self._bm25(text, depth), self._dense(text, depth)])[:k]
        raise ValueError(f"unknown mode {mode!r}")
```
- [ ] **Step 4:** Tests → PASS. **Step 5:** Commit `"Hybrid retrieval with RRF fusion"`.

---

### Task 3: Skill-coverage re-ranking

**Files:** Create `src/jobmatch/rerank.py`, `tests/test_rerank.py`.

**Interfaces — Produces:**
- `skill_coverage(resume_vecs: np.ndarray, job_vecs: np.ndarray, tau: float = 0.75) -> tuple[float, list[int]]`:
  the fraction of job skills whose best cosine to any resume skill is ≥ tau, plus the indices of **missing** job skills.
  Rows are L2-normalized skill-phrase embeddings. Empty job skills → `(0.0, [])`.
- `rerank(candidates: list[tuple[int, float]], coverage: dict[int, float], alpha: float = 0.5) -> list[tuple[int, float]]`:
  `score = alpha * rrf_norm + (1 - alpha) * coverage`, where `rrf_norm` = rrf score / max rrf score among the candidates.

- [ ] **Step 1: Failing tests**
```python
import numpy as np
from jobmatch.rerank import rerank, skill_coverage

def _n(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def test_coverage_soft_match_and_missing():
    resume = _n([[1, 0, 0], [0, 1, 0]])
    job = _n([[0.95, 0.05, 0], [0, 0, 1]])           # first ≈ resume skill 0, second unmatched
    cov, missing = skill_coverage(resume, job, tau=0.75)
    assert cov == 0.5 and missing == [1]

def test_coverage_empty():
    assert skill_coverage(_n([[1, 0]]), np.zeros((0, 2), np.float32)) == (0.0, [])
    assert skill_coverage(np.zeros((0, 2), np.float32), _n([[1, 0]])) == (0.0, [0])

def test_rerank_mixes_scores():
    cands = [(10, 0.03), (20, 0.02)]
    assert rerank(cands, {10: 0.0, 20: 1.0}, alpha=0.5)[0][0] == 20
    assert rerank(cands, {10: 0.0, 20: 1.0}, alpha=1.0)[0][0] == 10
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**
```python
"""Re-rank retrieved jobs by how many of the job's required skills the resume covers."""
from __future__ import annotations
import numpy as np

def skill_coverage(resume_vecs: np.ndarray, job_vecs: np.ndarray, tau: float = 0.75) -> tuple[float, list[int]]:
    if len(job_vecs) == 0:
        return 0.0, []
    if len(resume_vecs) == 0:
        return 0.0, list(range(len(job_vecs)))
    best = (job_vecs @ resume_vecs.T).max(axis=1)
    missing = [int(i) for i in np.flatnonzero(best < tau)]
    return 1.0 - len(missing) / len(job_vecs), missing

def rerank(candidates: list[tuple[int, float]], coverage: dict[int, float], alpha: float = 0.5) -> list[tuple[int, float]]:
    top = max((s for _, s in candidates), default=0.0) or 1.0
    scored = [(d, alpha * s / top + (1 - alpha) * coverage.get(d, 0.0)) for d, s in candidates]
    return sorted(scored, key=lambda x: -x[1])
```
- [ ] **Step 4:** Tests → PASS. **Step 5:** Commit `"Skill-coverage re-ranking"`.

---

### Task 4: JobIndex build / save / load

**Files:** Create `src/jobmatch/index.py`, `tests/test_index.py`.

**Interfaces — Consumes:** `build_corpus` output (Task 1), `bm25s`. **Produces:**
- `@dataclass JobIndex(jobs: pd.DataFrame, vectors: np.ndarray, bm25: bm25s.BM25, skills: list[list[dict]], skill_vecs: list[np.ndarray], manifest: dict)`
  - `skills[i]`: `[{"type": "SKILL"|"KNOWLEDGE", "text": str}]` for job i. `skill_vecs[i]`: an L2-normalized `[n_i, 384]` float16 array.
- `build_index(jobs, embed_docs: Callable[[list[str]], np.ndarray], extract_skills: Callable[[list[str]], list[list[dict]]], version: str) -> JobIndex`
- `JobIndex.save(path)` / `JobIndex.load(path) -> JobIndex`. Files: `jobs.parquet`, `vectors.npy` (float16), `bm25/`, `skills.json`,
  `skill_vecs.npz` (one concatenated array + offsets), `manifest.json` (`version, n_jobs, embed_model, skill_model, created`).
- Document text that gets embedded and indexed = `f"{title}\n{description}"`.

- [ ] **Step 1: Failing test:** build with fake callables (`embed_docs` returns random normalized vectors, `extract_skills`
  returns `[[{"type":"SKILL","text":"python"}]]*n`), save to `tmp_path`, load, then assert equal `jobs`, `np.allclose(vectors)`,
  `skills` round-trip, `skill_vecs[i].shape == (1, 384)`, and that BM25 retrieves the right doc for a title word.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (skill phrases are embedded with the same `embed_docs`; vectors are stored as float16 to halve size).
- [ ] **Step 4:** PASS. **Step 5:** Commit `"JobIndex build/save/load"`.

---

### Task 5: Kaggle build → HF dataset `srish327/tech-jobs-index` v1

**Files:** Create `notebooks/01_build_index.ipynb`, `DATA_CARD.md`.

- [ ] **Step 1:** Notebook cells:
  1. `pip install "jobmatch[index] @ git+https://github.com/MSR327/resume-job-matcher.git@<commit>"`.
  2. Load `/kaggle/input/linkedin-job-postings/postings.csv`; **print columns and assert** the 7 required ones exist.
  3. `jobs = build_corpus(df)`; print count, top titles, and description-length percentiles. Target ~10–20k. If it's outside that range, report and stop.
  4. `embed_docs` = `SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda").encode(..., normalize_embeddings=True, batch_size=128)`.
  5. `extract_skills` = `skillex.predict.SkillExtractor("srish327/deberta-v3-skillspan-crf", device="cuda")` over descriptions
     (dedupe skill texts per job, case-insensitive).
  6. `idx = build_index(...)`, `idx.save("artifacts/v1")`, then print sizes (must total < 200 MB) and 3 sample jobs with skills.
  7. Push `artifacts/v1` + `DATA_CARD.md` (as `README.md`) to the HF dataset repo `srish327/tech-jobs-index`, path `v1/`, using the
     Kaggle secret `hg`. **The user must attach the `hg` secret to this new notebook in the Add-ons → Secrets panel.**
- [ ] **Step 2:** `DATA_CARD.md`: source + link, **CC BY-SA 4.0** + attribution, filtering rules, counts, columns, LinkedIn-scrape provenance note, intended educational use.
- [ ] **Step 3:** Commit and push. Launch on Kaggle through a Sonnet subagent: import the notebook from the pinned raw URL, add the dataset `arshkon/linkedin-job-postings` as input, GPU T4, Internet On, then Save & Run All. Keep the notebook private.
- [ ] **Step 4:** Verify `https://huggingface.co/datasets/srish327/tech-jobs-index` has `v1/manifest.json`. Load it locally with `JobIndex.load`
  and run `HybridRetriever` on a sample resume. The top 5 titles should look sensible.

---

## Later milestones (detailed plans written when reached)
M4–5 evaluation (synthetic resumes, LLM judge on Kaggle with vLLM, human labels, ablations) · M6–7 `llm.py` / `explain.py` / citation check ·
M8 Streamlit + telemetry + fallback · M9 CI eval gate + Docker + deploy · M10 README / results / GIF.
