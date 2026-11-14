"""L2 eval: an LLM judge grades every (resume, job) in each configuration's top 10 -> nDCG@10.

    python scripts/eval_judge.py pairs [--n 60] [--split test|dev]   # top-10 per config -> data/eval/l2_pairs[_dev].json
    # Kaggle GPU: notebooks/02_judge.ipynb grades the pairs -> data/eval/judgments[_dev].jsonl (via the HF dataset repo)
    python scripts/eval_judge.py score [--split test|dev]             # judgments -> results/l2_judge[_dev].json

The test split is scored once. The dev split exists to choose the app configuration under L2 without looking at test.

The judge (IBM Granite-3.3-8B-Instruct, open weights, run on a free Kaggle GPU) is a different model family from
the resume writer (Qwen) and the app's explainer (gpt-oss), to limit self-preference. Groq's free tier
(200k tokens/day per model) could not grade ~1k pairs. Agreement with a human: scripts/agreement.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paths(split: str) -> tuple[Path, Path, str]:
    sfx = "" if split == "test" else f"_{split}"
    return Path(f"data/eval/l2_pairs{sfx}.json"), Path(f"data/eval/judgments{sfx}.jsonl"), f"results/l2_judge{sfx}.json"

CONFIGS = ("bm25", "dense", "hybrid", "bm25+rerank", "hybrid+rerank")


def make_pairs(n: int, split: str = "test") -> None:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from skillex.predict import SkillExtractor

    from jobmatch.embed import QUERY_PREFIX
    from jobmatch.index import JobIndex, clean_skills
    from jobmatch.rerank import rerank, skill_coverage
    from jobmatch.retrieve import HybridRetriever
    from jobmatch.synth import split_dev_test

    tuned = json.loads(Path("results/l1_retrieval.json").read_text())["tuned"]  # tau/alpha per base, from dev
    root = snapshot_download("srish327/tech-jobs-index", repo_type="dataset", allow_patterns=["v1/*"])
    idx = JobIndex.load(f"{root}/v1")
    items = [json.loads(x) for x in Path("data/eval/synth.jsonl").read_text().splitlines()]
    dev, test = split_dev_test(items, 100)
    queries = test[-n:] if split == "test" else dev[:n]  # end of test: disjoint from the CI-gate queries at the start

    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ext = SkillExtractor("srish327/deberta-v3-skillspan-crf")
    retr = HybridRetriever(idx.bm25, idx.vectors, lambda q: emb.encode(QUERY_PREFIX + q, normalize_embeddings=True))
    ids = idx.jobs["job_id"].tolist()
    out = []
    for q in queries:
        skills = sorted({s["text"] for s in clean_skills(ext.extract(q["resume"])["spans"])})
        rvec = emb.encode(skills, normalize_embeddings=True) if skills else np.zeros((0, 384))
        t = {"dense": [d for d, _ in retr.search(q["resume"], k=10, mode="dense")]}
        for base in ("bm25", "hybrid"):
            cands = retr.search(q["resume"], k=50, mode=base)
            t[base] = [d for d, _ in cands[:10]]
            tau, alpha = tuned[base]["tau"], tuned[base]["alpha"]
            cov = {d: skill_coverage(rvec, idx.job_skill_vecs(d), tau)[0] for d, _ in cands}
            t[f"{base}+rerank"] = [d for d, _ in rerank(cands, cov, alpha)[:10]]
        out.append({"resume_job_id": q["job_id"], "top10": {c: [ids[d] for d in t[c]] for c in CONFIGS}})
    pairs_path = paths(split)[0]
    pairs_path.write_text(json.dumps({"index_version": "v1", "split": split, "tuned": tuned, "queries": out}, indent=1))
    n_pairs = len({(q["resume_job_id"], j) for q in out for js in q["top10"].values() for j in js})
    print(f"{len(out)} resumes, {n_pairs} unique pairs -> {pairs_path}")


def _dcg(gains) -> float:
    return sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains[:10]))


def score(split: str = "test") -> None:
    pairs_path, judgments, out_path = paths(split)
    data = json.loads(pairs_path.read_text())
    grade, judge_model = {}, None
    for line in judgments.read_text().splitlines():
        r = json.loads(line)
        grade[(r["resume_job_id"], r["job_id"])] = r["grade"]
        judge_model = r.get("model", judge_model)
    res = {}
    for c in CONFIGS:
        nd, p10, unjudged = [], [], 0
        for q in data["queries"]:
            rid = q["resume_job_id"]
            gains = [grade.get((rid, j)) for j in q["top10"][c]]
            unjudged += sum(g is None for g in gains)
            gains = [g or 0 for g in gains]
            # Ideal = best grades among all jobs any configuration returned for this resume (shared pool).
            pool = {j for js in q["top10"].values() for j in js}
            ideal = sorted((grade.get((rid, j)) or 0 for j in pool), reverse=True)[:10]
            nd.append(_dcg(gains) / _dcg(ideal) if _dcg(ideal) else 0.0)
            p10.append(float(np.mean([g >= 2 for g in gains])))
        res[c] = {"nDCG@10": float(np.mean(nd)), "P@10(grade>=2)": float(np.mean(p10)), "unjudged": unjudged,
                  "per_query_ndcg": nd}
    src = [grade.get((q["resume_job_id"], q["resume_job_id"])) for q in data["queries"]]
    src = [g for g in src if g is not None]
    grades = [g for g in grade.values() if g is not None]
    other = [g for (rid, j), g in grade.items() if rid != j and g is not None]
    out = {"split": split, "n_resumes": len(data["queries"]), "n_pairs": len(grade), "judge": judge_model,
           "failed_judgments": sum(g is None for g in grade.values()),
           "configs": {c: {k: v for k, v in m.items() if k != "per_query_ndcg"} for c, m in res.items()},
           "source_job_grade_mean": float(np.mean(src)) if src else None, "n_source_judged": len(src),
           "other_job_grade_mean": float(np.mean(other)) if other else None,
           "grade_distribution": {g: grades.count(g) for g in range(4)}, "ci95": {}}
    rng = np.random.default_rng(0)
    for a, b in (("bm25", "bm25+rerank"), ("hybrid", "hybrid+rerank"), ("hybrid", "bm25"),
                 ("bm25+rerank", "hybrid+rerank")):
        x, y = np.array(res[a]["per_query_ndcg"]), np.array(res[b]["per_query_ndcg"])
        boot = (y - x)[rng.integers(0, len(x), size=(5000, len(x)))].mean(1)
        out["ci95"][f"{b} - {a}, nDCG@10"] = [float((y - x).mean()), float(np.percentile(boot, 2.5)),
                                              float(np.percentile(boot, 97.5))]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pairs", "score"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    args = ap.parse_args()
    make_pairs(args.n, args.split) if args.stage == "pairs" else score(args.split)


if __name__ == "__main__":
    main()
