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

**L2: LLM judge** (ibm-granite/granite-3.3-8b-instruct, 60 test resumes, 1507 judged pairs)

| configuration | nDCG@10 | P@10 (grade ≥ 2) |
|---|---|---|
| bm25 | 0.651 | 0.697 |
| dense | 0.661 | 0.722 |
| hybrid | 0.694 | 0.743 |
| bm25+rerank | 0.677 | 0.722 |
| hybrid+rerank | 0.705 | 0.743 |

Paired bootstrap (95% CI):

- bm25+rerank - bm25, nDCG@10: **+0.027** (-0.004 to +0.055)
- hybrid+rerank - hybrid, nDCG@10: **+0.011** (-0.012 to +0.032)
- bm25 - hybrid, nDCG@10: **-0.043** (-0.081 to -0.008)
- hybrid+rerank - bm25+rerank, nDCG@10: **+0.028** (-0.005 to +0.062)

Sanity check: the judge gives each resume's source job a mean grade of 2.36 / 3 (n = 53) vs. 1.90 for other retrieved jobs. Grades 0: 199, 1: 260, 2: 510, 3: 531; 7 unparseable outputs (truncated) count as 0.

**Explanations and serving** (10 test resumes × top 5, app pipeline, laptop CPU)

- citation pass rate **95%** (quotes found verbatim in the posting)
- missing-skill precision (lexical) **96%** (n = 304)
- LLM failure rate 0.0% (falls back to skill overlap)
- latency p50: skill extraction 0.46 s, retrieval 0.002 s, re-rank 0.09 s; LLM explanations 54 s (p95 65 s), almost all of it waiting on Groq's free-tier limit of 8k tokens/min, since one request needs ~8k tokens. Unthrottled, a full request took ~3 s.
- 8016 tokens and $0.0023 per request at list price

