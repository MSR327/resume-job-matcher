"""L1 eval: does the source job come back for its synthetic resume? Ablations + tuning.

    python scripts/eval_retrieval.py [--version v1]

BM25 / dense / hybrid / hybrid + skill re-rank. tau and alpha are tuned on the dev split only;
test is scored once per configuration, with a paired bootstrap CI for re-rank vs. hybrid.
Jobs with identical descriptions (reposts under another title) count as the same job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from skillex.predict import SkillExtractor

from jobmatch.index import JobIndex
from jobmatch.metrics import mrr, ngram_overlap, recall_at_k
from jobmatch.rerank import rerank
from jobmatch.retrieve import HybridRetriever
from jobmatch.synth import split_dev_test

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TAUS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def metrics(ranked, gold) -> dict:
    return {"R@1": recall_at_k(ranked, gold, 1), "R@5": recall_at_k(ranked, gold, 5),
            "R@10": recall_at_k(ranked, gold, 10), "MRR": mrr(ranked, gold), "n": len(gold)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of b - a with a 95% CI, resampling queries."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (b[idx] - a[idx]).mean(1)
    return float((b - a).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--synth", default="data/eval/synth.jsonl")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--out", default="results/l1_retrieval.json")
    args = ap.parse_args()

    root = snapshot_download("srish327/tech-jobs-index", repo_type="dataset", allow_patterns=[f"{args.version}/*"])
    idx = JobIndex.load(f"{root}/{args.version}")
    canon_of_desc: dict[str, int] = {}
    canon = [canon_of_desc.setdefault(hashlib.md5(d.lower().encode()).hexdigest(), i)
             for i, d in enumerate(idx.jobs["description"])]
    row_of = {j: i for i, j in enumerate(idx.jobs["job_id"])}

    items = [json.loads(line) for line in Path(args.synth).read_text().splitlines()]
    dev, test = split_dev_test(items, args.n_dev)
    print(f"{len(items)} resumes: dev {len(dev)}, test {len(test)}")

    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    ext = SkillExtractor("srish327/deberta-v3-skillspan-crf")
    retr = HybridRetriever(idx.bm25, idx.vectors,
                           lambda q: emb.encode(QUERY_PREFIX + q, normalize_embeddings=True))

    def run(split):
        out = []
        for it in split:
            t = {}
            t0 = time.perf_counter()
            skills = sorted({s["text"] for s in ext.extract(it["resume"])["spans"]})
            rvec = emb.encode(skills, normalize_embeddings=True) if skills else np.zeros((0, 384), np.float32)
            t["skills_ms"] = (time.perf_counter() - t0) * 1e3
            res = {}
            for mode in ("bm25", "dense", "hybrid"):
                t0 = time.perf_counter()
                res[mode] = retr.search(it["resume"], k=50, depth=100, mode=mode)
                t[f"{mode}_ms"] = (time.perf_counter() - t0) * 1e3
            # Best resume-skill similarity for every skill of every candidate: any tau is then cheap.
            cands = {d for m in ("bm25", "hybrid") for d, _ in res[m]}
            best = {d: (idx.job_skill_vecs(d).astype(np.float32) @ rvec.T).max(1) if len(rvec) and
                    len(idx.skill_ids[d]) else np.zeros(len(idx.skill_ids[d])) for d in cands}
            gold = canon[row_of[it["job_id"]]]
            out.append({"res": res, "best": best, "gold": gold, "t": t, "n_skills": len(skills),
                        "overlap4": ngram_overlap(it["resume"], idx.jobs["description"][row_of[it["job_id"]]])})
        return out

    def ranked_ids(q, mode, tau=None, alpha=None):
        cands = q["res"][mode]
        if tau is not None:
            cov = {d: float((q["best"][d] >= tau).mean()) if len(q["best"][d]) else 0.0 for d, _ in cands}
            cands = rerank(cands, cov, alpha)
        return [canon[d] for d, _ in cands]

    def score(qs, **kw):
        return metrics([ranked_ids(q, **kw) for q in qs], [q["gold"] for q in qs])

    print("running dev ...", flush=True)
    qd = run(dev)
    tuned, grids = {}, {}
    for base in ("bm25", "hybrid"):
        grid = {(t, a): score(qd, mode=base, tau=t, alpha=a) for t in TAUS for a in ALPHAS}
        (t, a), m = max(grid.items(), key=lambda kv: (kv[1]["MRR"], kv[1]["R@10"]))
        tuned[base], grids[base] = (t, a), grid
        print(f"dev: {base} MRR {score(qd, mode=base)['MRR']:.3f} | + re-rank tau={t} alpha={a} MRR {m['MRR']:.3f}")
    dev_table = {m: score(qd, mode=m) for m in ("bm25", "dense", "hybrid")}
    for base, (t, a) in tuned.items():
        dev_table[f"{base}+rerank"] = score(qd, mode=base, tau=t, alpha=a)
    choice = max(dev_table, key=lambda c: (dev_table[c]["MRR"], dev_table[c]["R@10"]))
    print(f"dev choice for the app: {choice}")

    print("running test ...", flush=True)
    qt = run(test)
    table = {m: score(qt, mode=m) for m in ("bm25", "dense", "hybrid")}
    for base, (t, a) in tuned.items():
        table[f"{base}+rerank"] = score(qt, mode=base, tau=t, alpha=a)

    def per_query(kind, k=None, **kw):
        vals = []
        for q in qt:
            r = ranked_ids(q, **kw)
            vals.append((1 / (r.index(q["gold"]) + 1) if q["gold"] in r else 0.0) if kind == "rr"
                        else float(q["gold"] in r[:k]))
        return np.array(vals)

    ci = {}
    for base, (t, a) in tuned.items():
        for metric, kind, k in (("MRR", "rr", None), ("R@10", "hit", 10)):
            ci[f"{base}+rerank - {base}, {metric}"] = paired_bootstrap(
                per_query(kind, k, mode=base), per_query(kind, k, mode=base, tau=t, alpha=a))
    ci["bm25 - hybrid, MRR"] = paired_bootstrap(per_query("rr", mode="hybrid"), per_query("rr", mode="bm25"))
    lat = {k: float(np.mean([q["t"][k] for q in qt])) for k in qt[0]["t"]}
    result = {
        "index_version": args.version, "n_dev": len(dev), "n_test": len(test),
        "tuned": {b: {"tau": t, "alpha": a} for b, (t, a) in tuned.items()}, "dev": dev_table, "dev_choice": choice,
        "test": table, "ci95": ci, "latency_ms_mean": lat,
        "overlap4_mean": float(np.mean([q["overlap4"] for q in qd + qt])),
        "resume_skills_mean": float(np.mean([q["n_skills"] for q in qd + qt])),
        "dev_grid_top5": {b: sorted(([f"tau={t},alpha={a}", m["MRR"]] for (t, a), m in g.items()),
                                    key=lambda x: -x[1])[:5] for b, g in grids.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"\n| config (test, n={len(test)}) | R@1 | R@5 | R@10 | MRR |\n|---|---|---|---|---|")
    for name, m in table.items():
        print(f"| {name} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.3f} |")
    for k, (d, lo, hi) in ci.items():
        print(f"{k}: {d:+.3f}  (95% CI {lo:+.3f} .. {hi:+.3f})")
    print("latency ms:", {k: round(v, 1) for k, v in lat.items()}, "| 4-gram overlap:",
          round(result["overlap4_mean"], 3))


if __name__ == "__main__":
    main()
