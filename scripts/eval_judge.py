"""L2 eval: an LLM judge grades every (resume, job) in each configuration's top 10 -> nDCG@10.

    python scripts/eval_judge.py [--n 30]

Unlike L1 (did the one source job come back?), this credits any good match. Judgments are
cached in data/eval/judgments.jsonl (resumable; Groq free tier is 1k requests/day per model).
The judge (gpt-oss-120b) is a different model family from the resume writer (qwen) to limit
self-preference. Its agreement with a human is measured separately (scripts/agreement.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from skillex.predict import SkillExtractor

from jobmatch.config import ALPHA, TAU
from jobmatch.embed import QUERY_PREFIX
from jobmatch.index import JobIndex, clean_skills
from jobmatch.judge import judge
from jobmatch.llm import GROQ_MODEL, groq, load_api_key
from jobmatch.rerank import rerank, skill_coverage
from jobmatch.retrieve import HybridRetriever
from jobmatch.synth import split_dev_test

CACHE = Path("data/eval/judgments.jsonl")
CONFIGS = ("bm25", "dense", "hybrid", "hybrid+rerank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="results/l2_judge.json")
    args = ap.parse_args()

    root = snapshot_download("srish327/tech-jobs-index", repo_type="dataset", allow_patterns=["v1/*"])
    idx = JobIndex.load(f"{root}/v1")
    items = [json.loads(x) for x in Path("data/eval/synth.jsonl").read_text().splitlines()]
    _, test = split_dev_test(items, 100)
    queries = test[-args.n:]  # the end of test: disjoint from the 50 CI-gate queries at the start

    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ext = SkillExtractor("srish327/deberta-v3-skillspan-crf")
    retr = HybridRetriever(idx.bm25, idx.vectors, lambda q: emb.encode(QUERY_PREFIX + q, normalize_embeddings=True))

    tops: dict[str, dict[str, list[int]]] = {}
    for q in queries:
        skills = sorted({s["text"] for s in clean_skills(ext.extract(q["resume"])["spans"])})
        rvec = emb.encode(skills, normalize_embeddings=True) if skills else np.zeros((0, 384))
        t = {m: [d for d, _ in retr.search(q["resume"], k=10, mode=m)] for m in ("bm25", "dense")}
        hy = retr.search(q["resume"], k=50, mode="hybrid")
        t["hybrid"] = [d for d, _ in hy[:10]]
        cov = {d: skill_coverage(rvec, idx.job_skill_vecs(d), TAU)[0] for d, _ in hy}
        t["hybrid+rerank"] = [d for d, _ in rerank(hy, cov, ALPHA)[:10]]
        tops[q["job_id"]] = t

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            cache[(r["resume_job_id"], r["job_id"])] = r["grade"]
    llm = groq(load_api_key(), model=GROQ_MODEL, rate_limit_retries=6, timeout=60)
    pairs = sorted({(q["job_id"], int(d)) for q in queries for t in tops[q["job_id"]].values() for d in t})
    jobs = idx.jobs
    todo = [(r, d) for r, d in pairs if (r, jobs["job_id"][d]) not in cache]
    print(f"{len(pairs)} unique pairs, {len(todo)} to judge", flush=True)
    resume_of = {q["job_id"]: q["resume"] for q in queries}
    with CACHE.open("a") as f:
        for n, (r, d) in enumerate(todo, 1):
            g = judge(llm, resume_of[r], jobs.iloc[d].to_dict())
            cache[(r, jobs["job_id"][d])] = g
            f.write(json.dumps({"resume_job_id": r, "job_id": jobs["job_id"][d], "grade": g,
                                "is_source": jobs["job_id"][d] == r, "model": GROQ_MODEL}) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"judged {n}/{len(todo)}", flush=True)

    res = {}
    for c in CONFIGS:
        nd, p10, missing = [], [], 0
        for q in queries:
            grades = [cache.get((q["job_id"], jobs["job_id"][d])) for d in tops[q["job_id"]][c]]
            missing += sum(g is None for g in grades)
            gains = [g or 0 for g in grades]
            # nDCG against the best grades seen for this resume across all configs (shared ideal).
            pool = [cache.get((q["job_id"], jobs["job_id"][d])) or 0 for d in
                    {d for t in tops[q["job_id"]].values() for d in t}]
            ideal = sorted(pool, reverse=True)[:10]
            nd.append(_dcg(gains) / _dcg(ideal) if _dcg(ideal) else 0.0)
            p10.append(np.mean([g >= 2 for g in gains]))
        res[c] = {"nDCG@10": float(np.mean(nd)), "P@10(grade>=2)": float(np.mean(p10)), "unjudged": missing}
    src = [cache.get((q["job_id"], q["job_id"])) for q in queries]
    src = [g for g in src if g is not None]
    grades = [g for g in cache.values() if g is not None]
    out = {"n_resumes": len(queries), "n_pairs": len(pairs), "judge": GROQ_MODEL, "configs": res,
           "source_job_grade_mean": float(np.mean(src)) if src else None, "n_source_judged": len(src),
           "grade_distribution": {g: grades.count(g) for g in range(4)}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n| config (judge: {GROQ_MODEL}, {len(queries)} resumes) | nDCG@10 | P@10 (grade>=2) |\n|---|---|---|")
    for c, m in res.items():
        print(f"| {c} | {m['nDCG@10']:.3f} | {m['P@10(grade>=2)']:.3f} |")
    print("source-job mean grade:", out["source_job_grade_mean"], "| grades:", out["grade_distribution"])


def _dcg(gains) -> float:
    return sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains[:10]))


if __name__ == "__main__":
    main()
