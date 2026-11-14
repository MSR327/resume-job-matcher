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

