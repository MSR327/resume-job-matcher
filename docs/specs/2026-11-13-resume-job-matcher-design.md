# Resume → Job Matcher: design

**Status:** approved 2026-11-13 · **Owner:** srishanth reddy · **Timebox:** ~10 working days

## 1. Goal

Paste a resume and get the 5 best-matching tech job postings from a ~15k-job corpus. For each match, show why it fits,
which required skills are missing, and verbatim quotes from the posting as evidence.

This is portfolio project 2. It must demonstrate **embeddings, hybrid retrieval, RAG with grounded citations, prompting,
LLM evaluation (with a human-validated judge), cost/latency tracking**, and the MLOps practices from project 1
(CI, eval gate, Docker, versioned artifacts). It reuses project 1's skill extractor
([`srish327/deberta-v3-skillspan-crf`](https://huggingface.co/srish327/deberta-v3-skillspan-crf), package `skillex`).

**Hard constraint:** $0 compute and hosting.

## 2. Architecture

Offline (Kaggle GPU, run once per index version):

1. `corpus`: load the job-postings dataset, filter to tech roles, clean, dedupe → `jobs.parquet`
2. `index`: for every job:
   - skills via `skillex` (DeBERTa+CRF)
   - dense vector via `BAAI/bge-small-en-v1.5` (384-d, normalized)
   - BM25 index (`bm25s`)
3. Publish the artifacts to the HF dataset repo `srish327/tech-jobs-index`, tagged `v1`, with a data card.

Online (Streamlit app, per request):

1. Extract resume skills (ONNX skill model, CPU).
2. Hybrid retrieval: BM25 top-100 and dense top-100, fused with Reciprocal Rank Fusion (k=60) → top 50.
3. Skill re-rank: `score = α · rrf_norm + (1 − α) · coverage`. `coverage` is the fraction of the job's extracted skills and knowledge
   matched by the resume's, using soft matching (cosine ≥ τ between bge embeddings of skill phrases). α and τ are tuned on the dev split of the synthetic eval set.
4. Explain the top 5 with the LLM (Groq free tier) → JSON `{fit_summary, matched_skills[], missing_skills[], evidence[{claim, quote}]}`.
5. Citation check: a quote must occur verbatim in the posting (whitespace/case-normalized), otherwise it is dropped.
   Report the citation pass rate.
6. If the LLM fails (rate limit, timeout, bad JSON): retry once, then show the ranked jobs and skill overlap without explanations.

### Modules (`src/jobmatch/`)
| Module | Responsibility | Depends on |
|---|---|---|
| `corpus.py` | load / filter / clean / dedupe postings | pandas |
| `index.py` | build + save + load BM25, vectors, job skills | bm25s, sentence-transformers, skillex |
| `retrieve.py` | BM25, dense, RRF fusion | index |
| `rerank.py` | soft skill coverage, final score | embeddings |
| `llm.py` | `LLM` protocol + `GroqLLM`, `VLLMLLM` (Kaggle), `FakeLLM` (tests) | httpx |
| `explain.py` | prompt, JSON parsing/repair, citation check | llm |
| `telemetry.py` | per-request JSON log: stage latencies, tokens, est. cost, citations | stdlib |
| `app.py` (`streamlit_app.py`) | UI + Stats page | all |

## 3. Data

- **Jobs:** [LinkedIn Job Postings (2023–2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by arshkon,
  about 124k postings with full `description` text, **CC BY-SA 4.0** (checked 2026-11-13 via the Kaggle API).
  Filtered by title/description to tech roles (software, data, ML, DevOps, cloud, security, QA), ~15k after dedup.
  - **Share-alike:** the derived index (`srish327/tech-jobs-index`: filtered text, skills, embeddings) is published under
    **CC BY-SA 4.0** with attribution to the source dataset. Code stays MIT/Apache (code is not an adaptation of the data).
  - The source was scraped from LinkedIn. The data card says so, and the project is non-commercial and educational.
  - Rejected alternatives: `lukebarousse/data_jobs` (Apache-2.0 but has no description text), `datastax/linkedin_job_listings`
    (no license shown), `xanderios/linkedin-job-postings` (MIT but an empty card, so provenance is unclear; kept as a fallback).
- **Resumes:** no real people's resumes are stored or published.
  - Synthetic resumes (LLM-written, see §4).
  - A public resume dataset with a permissive license, for realistic inputs (license checked).
  - User input in the demo is processed in memory and never logged (telemetry logs numbers, never text).

## 4. Evaluation

| Layer | What | Metric |
|---|---|---|
| L1 synthetic | 300 jobs → the LLM writes a fitting resume, paraphrased (no copied 4-grams). Split 100 dev / 200 test | Recall@10, MRR of the source job; report n-gram overlap |
| L2 LLM judge | ~100 realistic resumes × top-10 → open LLM on Kaggle grades 0–3 | nDCG@10 |
| L3 human check | the owner labels 50 resume–job pairs (0–3) | judge–human agreement: weighted Cohen's κ, exact %, |Δ| |
| Explanations | all demo/eval explanations | citation pass rate; missing-skill precision (missing skill really absent from the resume) |

**Ablations:** BM25 · dense · hybrid · hybrid + skill re-rank · hybrid + LLM re-rank (open 7B on Kaggle, listwise),
each with latency and estimated cost per request. Tuning uses dev only; test is evaluated once per configuration.

## 5. MLOps

- **CI (GitHub Actions):** ruff, unit tests (no network, `FakeLLM`), docker build, **eval gate**: retrieval and re-ranking on a frozen
  50-query set over a 2k-job slice (shipped as a small fixture/artifact). Fails below the Recall@10 / nDCG thresholds. No LLM calls in CI.
- **Versioning:** the index is versioned in the HF dataset repo, and the app pins a version.
- **Deploy:** Streamlit Community Cloud from `main`. `GROQ_API_KEY` is set by the owner in the app's Secrets. Also a Dockerfile for local use.
- **Monitoring:** JSON-lines telemetry, plus a Stats page with p50/p95 latency per stage, tokens, $/request (at public list prices), and citation pass rate.

## 6. Out of scope

User accounts or stored resumes, live scraping, LLM fine-tuning, non-English or non-tech jobs, a vector-database server
(in-memory NumPy/FAISS is enough at 15k docs; the README notes when to switch).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Dataset license disallows reuse | Check on day 1; switch dataset / publish derived features only |
| Groq free-tier limits | Bulk eval runs on Kaggle with vLLM; Groq only for the demo plus a small sample |
| Streamlit 1 GB RAM | ONNX skill model (271 MB) + bge-small (~130 MB) + index (~50 MB); otherwise precompute more |
| Synthetic resumes leak job wording | Paraphrase prompt plus an n-gram overlap filter; report overlap |
| LLM judge disagrees with the human | Report κ honestly; if low, lean on L1 and the human labels |

## 8. Milestones

1. Data + license + data card · 2. Offline index on Kaggle → HF `v1` · 3. Retrieval + re-rank + tests ·
4–5. Eval sets + ablations · 5. Human labels + judge agreement · 6–7. LLM explanations + citation check ·
8. Streamlit app + Stats + fallback · 9. CI + eval gate + Docker + deploy · 10. README, results, demo GIF.
