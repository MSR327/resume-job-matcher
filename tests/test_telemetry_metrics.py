import math

import pytest

from jobmatch.metrics import mrr, ndcg_at_k, ngram_overlap, recall_at_k, weighted_kappa
from jobmatch.telemetry import Telemetry, summarize


def test_recall_and_mrr():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["q"]]
    gold = ["b", "z", "nope"]
    assert recall_at_k(ranked, gold, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, gold, 3) == pytest.approx(2 / 3)
    assert mrr(ranked, gold) == pytest.approx((1 / 2 + 1 / 3 + 0) / 3)


def test_ndcg():
    assert ndcg_at_k([3, 2, 0], 3) == pytest.approx(1.0)
    worse = ndcg_at_k([0, 2, 3], 3)
    assert 0 < worse < 1
    assert ndcg_at_k([0, 0], 2) == 0.0


def test_weighted_kappa():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0]) < 0
    # off-by-one disagreements are penalized less than off-by-three (quadratic weights)
    near = weighted_kappa([0, 1, 2, 3, 0, 3], [1, 1, 2, 2, 0, 3])
    far = weighted_kappa([0, 1, 2, 3, 0, 3], [3, 1, 2, 0, 0, 3])
    assert near > far


def test_ngram_overlap():
    job = "experience with python and docker is required for this role"
    assert ngram_overlap("i know python and docker is required for", job, n=4) > 0.5
    assert ngram_overlap("built ml pipelines in pytorch", job, n=4) == 0.0


def test_telemetry_logs_numbers_only(tmp_path):
    t = Telemetry(tmp_path / "t.jsonl")
    with t.request() as rec:
        with rec.stage("retrieve"):
            pass
        rec.add(prompt_tokens=100, completion_tokens=20, cost_usd=0.001, citations_total=4, citations_kept=3)
        with pytest.raises(TypeError):
            rec.add(resume="some text")  # strings are refused: no resume text can leak into logs
    rows = t.read()
    assert len(rows) == 1 and "retrieve_s" in rows[0] and rows[0]["citations_kept"] == 3
    s = summarize(rows)
    assert s["n"] == 1 and s["citation_pass_rate"] == pytest.approx(0.75)
    assert not math.isnan(s["latency_p95"]["retrieve_s"])
