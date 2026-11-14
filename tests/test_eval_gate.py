"""CI eval gate: real retrieval + RRF + skill re-rank on a frozen 50-query / 2,000-job fixture.

No models or network: query vectors and skill coverage are precomputed (scripts/make_ci_fixture.py).
Fails if a code change makes retrieval worse than the recorded floor.
"""

import json
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import pytest

from jobmatch.config import ALPHA
from jobmatch.index import doc_text
from jobmatch.metrics import mrr, recall_at_k
from jobmatch.rerank import rerank
from jobmatch.retrieve import HybridRetriever

FIX = Path(__file__).parent / "fixtures" / "eval_gate"
# Floors: a little under the scores measured when the fixture was frozen (see README "Evaluation").
FLOORS = json.loads((FIX / "floors.json").read_text()) if (FIX / "floors.json").exists() else {}

pytestmark = pytest.mark.skipif(not (FIX / "arrays.npz").exists(), reason="fixture not built")


def _scores():
    jobs = pd.read_parquet(FIX / "jobs.parquet")
    arr = np.load(FIX / "arrays.npz")
    q = json.loads((FIX / "queries.json").read_text())
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(doc_text(jobs), stopwords="en", show_progress=False), show_progress=False)
    qv = {t: v.astype(np.float32) for t, v in zip(q["texts"], arr["query_vecs"])}
    retr = HybridRetriever(bm25, arr["job_vecs"], lambda text: qv[text])
    ids = jobs["job_id"].tolist()
    out = {}
    for mode in ("bm25", "dense", "hybrid", "rerank"):
        ranked = []
        for qi, text in enumerate(q["texts"]):
            cands = retr.search(text, k=50, depth=100, mode="hybrid" if mode == "rerank" else mode)
            if mode == "rerank":
                cands = rerank(cands, {d: float(arr["coverage"][qi, d]) for d, _ in cands}, ALPHA)
            ranked.append([ids[d] for d, _ in cands])
        out[mode] = {"R@10": recall_at_k(ranked, q["gold_job_ids"], 10), "MRR": mrr(ranked, q["gold_job_ids"])}
    return out


def test_retrieval_meets_floors():
    scores = _scores()
    print(json.dumps(scores, indent=1))
    for mode, floor in FLOORS.items():
        for metric, value in floor.items():
            assert scores[mode][metric] >= value, f"{mode} {metric} {scores[mode][metric]:.3f} < floor {value}"
