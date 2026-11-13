import numpy as np

from jobmatch.rerank import rerank, skill_coverage


def _n(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_coverage_soft_match_and_missing():
    resume = _n([[1, 0, 0], [0, 1, 0]])
    job = _n([[0.95, 0.05, 0], [0, 0, 1]])  # first ≈ resume skill 0, second unmatched
    cov, missing = skill_coverage(resume, job, tau=0.75)
    assert cov == 0.5 and missing == [1]


def test_coverage_empty():
    assert skill_coverage(_n([[1, 0]]), np.zeros((0, 2), np.float32)) == (0.0, [])
    assert skill_coverage(np.zeros((0, 2), np.float32), _n([[1, 0]])) == (0.0, [0])


def test_rerank_mixes_scores():
    cands = [(10, 0.03), (20, 0.02)]
    assert rerank(cands, {10: 0.0, 20: 1.0}, alpha=0.5)[0][0] == 20
    assert rerank(cands, {10: 0.0, 20: 1.0}, alpha=1.0)[0][0] == 10
