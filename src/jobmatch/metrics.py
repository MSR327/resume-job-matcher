"""Evaluation metrics: retrieval (Recall@k, MRR, nDCG), judge agreement (weighted kappa), leakage."""

from __future__ import annotations

import math
import re


def recall_at_k(ranked: list[list[str]], gold: list[str], k: int) -> float:
    """Share of queries whose single gold job appears in the top k."""
    return sum(g in r[:k] for r, g in zip(ranked, gold)) / len(gold)


def mrr(ranked: list[list[str]], gold: list[str]) -> float:
    return sum(1 / (r.index(g) + 1) if g in r else 0.0 for r, g in zip(ranked, gold)) / len(gold)


def ndcg_at_k(gains: list[float], k: int = 10) -> float:
    """nDCG of one ranked list of graded relevances (e.g. judge scores 0-3), ideal = same grades sorted."""
    def dcg(g):
        return sum((2**x - 1) / math.log2(i + 2) for i, x in enumerate(g[:k]))

    ideal = dcg(sorted(gains, reverse=True))
    return dcg(gains) / ideal if ideal else 0.0


def weighted_kappa(a: list[int], b: list[int], n_classes: int = 4) -> float:
    """Cohen's kappa with quadratic weights: being off by 2 grades costs 4x being off by 1."""
    n = len(a)
    obs = [[0.0] * n_classes for _ in range(n_classes)]
    for x, y in zip(a, b):
        obs[x][y] += 1 / n
    pa = [sum(row) for row in obs]
    pb = [sum(obs[i][j] for i in range(n_classes)) for j in range(n_classes)]
    w = [[(i - j) ** 2 / (n_classes - 1) ** 2 for j in range(n_classes)] for i in range(n_classes)]
    o = sum(w[i][j] * obs[i][j] for i in range(n_classes) for j in range(n_classes))
    e = sum(w[i][j] * pa[i] * pb[j] for i in range(n_classes) for j in range(n_classes))
    return 1 - o / e if e else 1.0


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def ngram_overlap(resume: str, job: str, n: int = 4) -> float:
    """Share of the resume's n-grams that also occur in the job: high = the synthetic resume copied it."""
    r = _ngrams(resume, n)
    return len(r & _ngrams(job, n)) / len(r) if r else 0.0
