"""Freeze a small, model-free eval fixture for the CI gate.

    python scripts/make_ci_fixture.py

50 synthetic test resumes + a 2,000-job slice of index v1 (their source jobs + random others).
Stores what CI needs without downloading models: job texts (for BM25), job vectors, the
resume (query) vectors, and the resume-skill coverage of every job at the configured TAU.
CI then runs the real retrieval + RRF + re-rank code and fails below the recorded thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from skillex.predict import SkillExtractor

from jobmatch.config import TAU
from jobmatch.embed import QUERY_PREFIX
from jobmatch.index import JobIndex, clean_skills
from jobmatch.rerank import skill_coverage
from jobmatch.synth import split_dev_test

OUT = Path("tests/fixtures/eval_gate")
N_QUERIES, N_JOBS, SEED = 50, 2000, 0


def main() -> None:
    root = snapshot_download("srish327/tech-jobs-index", repo_type="dataset", allow_patterns=["v1/*"])
    idx = JobIndex.load(f"{root}/v1")
    items = [json.loads(x) for x in Path("data/eval/synth.jsonl").read_text().splitlines()]
    _, test = split_dev_test(items, 100)
    queries = test[:N_QUERIES]  # from the test split, never used for tuning

    row = {j: i for i, j in enumerate(idx.jobs["job_id"])}
    gold = sorted({row[q["job_id"]] for q in queries})
    rng = np.random.default_rng(SEED)
    others = rng.choice(np.setdiff1d(np.arange(len(idx.jobs)), gold), N_JOBS - len(gold), replace=False)
    rows = np.sort(np.concatenate([gold, others]))

    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ext = SkillExtractor("srish327/deberta-v3-skillspan-crf")
    qvec = emb.encode([QUERY_PREFIX + q["resume"] for q in queries], normalize_embeddings=True)
    cov = np.zeros((len(queries), len(rows)), np.float16)
    for qi, q in enumerate(queries):
        skills = sorted({s["text"] for s in clean_skills(ext.extract(q["resume"])["spans"])})
        rvec = emb.encode(skills, normalize_embeddings=True) if skills else np.zeros((0, 384))
        cov[qi] = [skill_coverage(rvec, idx.job_skill_vecs(d), TAU)[0] for d in rows]

    OUT.mkdir(parents=True, exist_ok=True)
    idx.jobs.iloc[rows][["job_id", "title", "description"]].to_parquet(OUT / "jobs.parquet", compression="zstd",
                                                                       index=False)
    np.savez_compressed(OUT / "arrays.npz", job_vecs=idx.vectors[rows], query_vecs=qvec.astype(np.float16),
                        coverage=cov)
    (OUT / "queries.json").write_text(json.dumps(
        {"gold_job_ids": [q["job_id"] for q in queries], "tau": TAU, "index_version": "v1",
         "texts": [q["resume"] for q in queries]}, indent=1))
    print({f.name: f"{f.stat().st_size / 1e6:.2f} MB" for f in OUT.iterdir()})


if __name__ == "__main__":
    main()
