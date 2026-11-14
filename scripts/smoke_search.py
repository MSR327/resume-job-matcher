"""End-to-end smoke test against the published index: resume -> hybrid search -> skill re-rank.

    python scripts/smoke_search.py [--version v1] [--mode hybrid]
"""

from __future__ import annotations

import argparse

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from jobmatch.index import JobIndex
from jobmatch.rerank import rerank, skill_coverage
from jobmatch.retrieve import HybridRetriever

RESUME = """Machine learning engineer, 2 years. Fine-tuned transformer models (BERT, DeBERTa) for NER in PyTorch
and Hugging Face. Built FastAPI services in Docker, CI/CD with GitHub Actions, experiment tracking.
Python, SQL, pandas. Deployed models with ONNX. Mentored interns and presented results to stakeholders."""
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    args = ap.parse_args()

    root = snapshot_download("srish327/tech-jobs-index", repo_type="dataset", allow_patterns=[f"{args.version}/*"])
    idx = JobIndex.load(f"{root}/{args.version}")
    print(idx.manifest)
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    r = HybridRetriever(idx.bm25, idx.vectors, lambda q: emb.encode(QUERY_PREFIX + q, normalize_embeddings=True))
    cands = r.search(RESUME, k=50, mode=args.mode)

    from skillex.predict import SkillExtractor

    resume_skills = sorted({s["text"] for s in SkillExtractor("srish327/deberta-v3-skillspan-crf").extract(RESUME)["spans"]})
    print("\nresume skills:", resume_skills)
    rvec = emb.encode(resume_skills, normalize_embeddings=True) if resume_skills else np.zeros((0, 384))
    cov = {d: skill_coverage(rvec, idx.job_skill_vecs(d))[0] for d, _ in cands}
    for name, ranked in [("retrieval only", cands), ("+ skill re-rank", rerank(cands, cov))]:
        print(f"\n== {name} ==")
        for d, s in ranked[:5]:
            j = idx.jobs.iloc[d]
            print(f"  {s:.3f}  cov={cov[d]:.2f}  {j['title']} @ {j['company']}")


if __name__ == "__main__":
    main()
