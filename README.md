# Resume → Job Matcher

Paste a resume, get the best-matching tech jobs from **7,817 real LinkedIn postings** with the skills you have,
the ones you're missing, and **quotes from each posting as evidence**. Every quote is checked against the posting,
so the explanation can't invent requirements.

**Live demo:** TBD · **Job index:** [srish327/tech-jobs-index](https://huggingface.co/datasets/srish327/tech-jobs-index) ·
**Skill model:** [srish327/deberta-v3-skillspan-crf](https://huggingface.co/srish327/deberta-v3-skillspan-crf)
(from [project 1](https://github.com/MSR327/skillspan-ner))

[![ci](https://github.com/MSR327/resume-job-matcher/actions/workflows/ci.yml/badge.svg)](https://github.com/MSR327/resume-job-matcher/actions/workflows/ci.yml)

## How it works

```mermaid
flowchart LR
  subgraph offline["Offline · Kaggle T4, ~30 min per index version"]
    A[LinkedIn postings 124k] --> B[tech-role filter + dedupe<br/>7,817 jobs]
    B --> C[bge-small embeddings]
    B --> D[BM25 index]
    B --> E[skills: DeBERTa-v3 + CRF<br/>299k spans, 152k distinct]
    C & D & E --> F[(HF dataset<br/>tech-jobs-index v1)]
  end
  subgraph online["Online · Streamlit, CPU only, no PyTorch"]
    R[resume] --> S[skills: ONNX DeBERTa]
    R --> H[BM25 top-100 + dense top-100<br/>Reciprocal Rank Fusion]
    F --> H
    H --> K[skill-coverage re-rank]
    S --> K
    K --> L[LLM explains top 5<br/>gpt-oss-120b on Groq]
    L --> V[citation check:<br/>quotes must be verbatim]
  end
```

1. **Retrieve:** BM25 (exact tool names like "Terraform") and bge-small embeddings (meaning, e.g. "built ML pipelines" ≈
   "MLOps"), fused with Reciprocal Rank Fusion (k = 60).
2. **Re-rank:** `score = α · retrieval + (1 − α) · coverage`, where coverage is the share of the job's extracted skills that
   some resume skill matches (cosine ≥ τ between skill-phrase embeddings). α and τ are tuned on a dev split.
3. **Explain:** the LLM returns JSON `{fit_summary, matched_skills, missing_skills, evidence[{claim, quote}]}` for the top 5,
   in parallel. Quotes not found verbatim in the posting (after normalizing whitespace, case and typography) are dropped.
   If the LLM fails (rate limit, timeout, bad JSON) it retries once, then the app shows the ranked jobs with skill overlap only.

## Results

**Headline:** on held-out synthetic resumes, **keyword search (BM25) + skill-coverage re-ranking** finds the right job in
the top 10 **77%** of the time (MRR 0.57). Re-ranking by skill coverage adds **+5.1 points Recall@10** over BM25 alone
(95% CI +0.7 to +10.2). Explanations quote the posting verbatim **95%** of the time.

**Finding: dense retrieval hurt.** bge-small sees only the first 512 tokens of a posting, which are mostly company boilerplate,
and fusing it with BM25 lowered MRR by 13 points versus BM25 alone. Chunked max-sim embeddings helped dense on its own
(R@10 0.50 → 0.57 on a 30-resume preview of dev) but not the fusion, so the app ships BM25 + re-rank (chosen on dev, not test). Caveat: synthetic
resumes are written from one posting, which favors exact tool names; L2 checks whether that holds for "any good match".

**L1: synthetic resumes** (test n = 137; tuned on dev n = 100; mean 4-gram overlap with the source posting 0.7%)

| configuration | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.438 | 0.693 | 0.723 | 0.546 |
| dense | 0.204 | 0.328 | 0.431 | 0.272 |
| hybrid | 0.299 | 0.540 | 0.657 | 0.416 |
| bm25+rerank ← app | 0.453 | 0.730 | 0.774 | 0.574 |
| hybrid+rerank | 0.387 | 0.606 | 0.715 | 0.489 |

Paired bootstrap (95% CI, test):

- bm25+rerank - bm25, MRR: **+0.028** (-0.002 to +0.061)
- bm25+rerank - bm25, R@10: **+0.051** (+0.007 to +0.102)
- hybrid+rerank - hybrid, MRR: **+0.073** (+0.018 to +0.128)
- hybrid+rerank - hybrid, R@10: **+0.058** (+0.007 to +0.117)
- bm25 - hybrid, MRR: **+0.131** (+0.067 to +0.194)

Tuned on dev: bm25: τ=0.85, α=0.2, hybrid: τ=0.85, α=0.3. App configuration (best dev MRR): `bm25+rerank`.

**Explanations and serving** (10 test resumes × top 5, app pipeline, laptop CPU)

- citation pass rate **95%** (quotes found verbatim in the posting)
- missing-skill precision (lexical) **96%** (n = 304)
- LLM failure rate 0.0% (falls back to skill overlap)
- latency p50: skill extraction 0.46 s, retrieval 0.002 s, re-rank 0.09 s; LLM explanations 54 s (p95 65 s), almost all of it waiting on Groq's free-tier limit of 8k tokens/min, since one request needs ~8k tokens. Unthrottled, a full request took ~3 s.
- 8016 tokens and $0.0023 per request at list price

Notes: the explanation eval used 5 explanations per request with 6,000-character posting context. After seeing the
free-tier token limit, the app explains the top 3 with 4,000 characters (~4k tokens per request) so that about two
requests per minute get explanations; the rest fall back to ranked jobs with skill overlap.

## Evaluation design

| Layer | Question | Method |
|---|---|---|
| L1 synthetic | Does the right job come back? | For 300 sampled jobs, an LLM (qwen3.8-27b) writes a fitting resume **in its own words** (resumes sharing >10% of 4-grams with the posting are rejected). Recall@k / MRR of the source job. Split 100 dev (tuning) / 200 test (scored once). Reposts with identical text count as the same job. |
| L2 LLM judge | Are the top 10 good matches? | IBM Granite-3.3-8B-Instruct (open weights, free Kaggle T4s) grades every (resume, job) pair in each configuration's top 10, 0–3 with a written rubric → nDCG@10. 60 test resumes, 1,507 pairs. The judge is a different model family from the resume writer (Qwen) and the explainer (gpt-oss), to limit self-preference. |
| L3 human check | Can the judge be trusted? | 50 pairs, sampled evenly across the judge's grades, labeled blind by me with the same rubric → quadratic-weighted Cohen's κ. |
| Explanations | Are they grounded? | Citation pass rate, missing-skill precision, latency and $/request through the exact app pipeline. |

CI runs an **eval gate**: the real retrieval + RRF + re-rank code on a frozen fixture (50 test queries, 2,000-job slice,
precomputed vectors, no models or network) and fails if Recall@10 or MRR drops below the recorded floor.

## Engineering notes (what broke and what it taught me)

- **Index size, 309 MB → 179 MB.** The first full build failed its own size check. Each job stored its own copy of every
  skill vector ("Python" ×2,000). Storing each of the 152k distinct phrases once, plus per-job ids, fixed it. The same
  investigation showed one-word SKILL spans ("skilled", "motivated") were noise that would have lowered every coverage
  score, so they are filtered (14% of spans).
- **Fail fast in the build notebook.** That failure came after 32 GPU-minutes and threw the result away. The notebook
  now checks the HF token and write access in its first cell, saves before any size check, and only warns on size.
- **GPU OOM on long postings.** All sentences of a 17k-character posting went through DeBERTa in one batch; fixed
  upstream in `skillex` with mini-batches trimmed to each batch's longest window.
- **No PyTorch in the app.** Both models run with ONNX Runtime: the embedder matches sentence-transformers at cosine
  1.0000, and the skill extractor agrees with PyTorch on 99.5% of spans. ONNX Runtime's constant folding turned the
  fp16-stored weights back into a full fp32 copy (+250 MB), so the app disables graph optimization (`low_memory=True`).
  The app runs in ~970 MB.
- **Hosted model retired.** The first synthetic-data run hit 404s: the Groq model in my config no longer existed, and the
  loop silently skipped every job. Now a circuit breaker stops after 15 failures in a row.
- **Free-tier LLM limits shaped the evaluation.** Groq's free tier allows 200k tokens/day and 8k tokens/minute per model
  (and 1k output tokens/minute for some models), learned from 429 responses. Synthetic-resume generation stopped at 237
  (100 dev / 137 test) when the daily cap hit, and the LLM judge moved to an open model on a free Kaggle GPU, where
  ~1.5k judgments cost nothing. One explained app request uses ~4–8k tokens, so the app explains only the top 3.
- **Typography vs. citations.** gpt-oss writes non-breaking hyphens (`end‑to‑end`), which failed exact-quote checks for
  reasons unrelated to grounding; the check normalizes them.

## Run it

```bash
pip install -r requirements.txt
echo 'GROQ_API_KEY=...' > .env      # optional: without it the app shows ranked jobs without explanations
streamlit run streamlit_app.py
```

Or with Docker: `docker build -t jobmatch . && docker run -p 8501:8501 -e GROQ_API_KEY=... jobmatch`.

Reproduce the evaluation (needs `pip install -e ".[index]" sentence-transformers` and a Groq key):

```bash
python scripts/make_synth.py --n 300     # synthetic resumes (resumable; stopped at 237 by the daily token cap)
python scripts/eval_retrieval.py         # L1 + tuning of tau/alpha on dev -> results/l1_retrieval.json
python scripts/eval_judge.py pairs       # L2: top-10 of each config -> data/eval/l2_pairs.json
# run notebooks/02_judge.ipynb on a Kaggle GPU (uploads judgments to the HF dataset repo), then:
python scripts/eval_judge.py score       # nDCG@10 per config -> results/l2_judge.json
streamlit run tools/label_app.py         # L3 blind human labels, then:
python scripts/agreement.py
python scripts/eval_explain.py --n 10    # citations, missing-skill precision, latency, cost
python scripts/report.py                 # results/*.json -> the tables above
```

## Data and privacy

- Jobs: [LinkedIn Job Postings 2023–24](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by arshkon,
  CC BY-SA 4.0. The derived index is published under the same license ([data card](DATA_CARD.md)).
- No real resumes are stored or published. Eval resumes are synthetic. In the app, the resume is processed in memory
  and never logged; telemetry records numbers only (a test enforces that strings are rejected).

## Limitations

- Seniority is not modeled: a 2-year resume can rank "Lead" roles highly.
- Skill spans from job postings are long and fuzzy (skill F1 ~0.56 for the extractor), so coverage is a rough signal.
- Synthetic resumes are cleaner than real ones; L1 numbers are an upper-bound-ish sanity check, which is why L2/L3 exist.
- The corpus is a 2023–24 snapshot, US-centric, tech roles only.

## Repo map

| Path | What |
|---|---|
| `src/jobmatch/` | corpus filter, index, retrieval, re-rank, LLM client, explanations, judge, metrics, telemetry, ONNX embedder, pipeline |
| `streamlit_app.py`, `pages/` | app + Stats page |
| `notebooks/01_build_index.ipynb` | Kaggle build of the index |
| `notebooks/02_judge.ipynb` | Kaggle L2 judge (Granite-3.3-8B) |
| `scripts/` | eval scripts, smoke test, CI fixture builder, parity checks |
| `tools/label_app.py` | blind labeling UI |
| `tests/` | unit tests + eval gate (no network) |
