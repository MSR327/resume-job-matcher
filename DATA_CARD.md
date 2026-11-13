---
license: cc-by-sa-4.0
language: [en]
tags: [job-postings, skill-extraction, retrieval, rag]
size_categories: [10K<n<100K]
---

# tech-jobs-index

A searchable index of **English tech job postings**, used by
[resume-job-matcher](https://github.com/MSR327/resume-job-matcher): job text, BM25 index, dense embeddings,
and skills/knowledge extracted with [`srish327/deberta-v3-skillspan-crf`](https://huggingface.co/srish327/deberta-v3-skillspan-crf).

## Source and license
- Derived from **[LinkedIn Job Postings (2023–2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)** by arshkon,
  licensed **CC BY-SA 4.0**. This derived dataset is released under the same license (share-alike), with attribution to the original.
- The source postings were collected from LinkedIn by the original dataset author. This index is for **non-commercial, educational** use.

## How it was built (`v1/`)
1. Kept postings whose title matches tech roles (software, data, ML/AI, DevOps/SRE/cloud, security, QA, databases),
   excluding sales/recruiting/non-software engineering titles. Descriptions under 400 characters were dropped.
2. Removed reposts (same company + title + description) and normalized whitespace and bullets.
3. Embedded `title + description` with `BAAI/bge-small-en-v1.5` (normalized, stored float16), then built a BM25 index (`bm25s`).
4. Extracted SKILL/KNOWLEDGE spans from each description, deduplicated them per job (case-insensitive), and embedded each phrase.

Exact counts and the build commit are in `v1/manifest.json`.

## Files (`v1/`)
| File | Content |
|---|---|
| `jobs.parquet` | `job_id, title, company, location, description, url, listed_at` |
| `vectors.npy` | `[n_jobs, 384]` float16 job embeddings |
| `bm25/` | `bm25s` index over the same texts |
| `skills.json` | per job: list of `{type, text}` |
| `skill_vecs.npz` | skill-phrase embeddings (`flat` + `offsets`) |
| `manifest.json` | version, counts, models, build commit |

## Limitations
Postings are mostly US and from 2023–24. Skill extraction is imperfect: span F1 is about 0.63 exact and 0.82 lenient on SkillSpan.
The title filter is rule-based and will miss some roles or include some borderline ones.
