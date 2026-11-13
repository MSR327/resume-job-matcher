"""Re-rank retrieved jobs by how many of the job's required skills the resume covers."""

from __future__ import annotations

import numpy as np


def skill_coverage(resume_vecs: np.ndarray, job_vecs: np.ndarray, tau: float = 0.75) -> tuple[float, list[int]]:
    """Share of job skills with a resume skill at cosine >= tau (soft match), plus the missing ones.

    Rows are L2-normalized skill-phrase embeddings, so "ML pipelines" can count toward "MLOps".
    """
    if len(job_vecs) == 0:
        return 0.0, []
    if len(resume_vecs) == 0:
        return 0.0, list(range(len(job_vecs)))
    best = (job_vecs.astype(np.float32) @ resume_vecs.astype(np.float32).T).max(axis=1)
    missing = [int(i) for i in np.flatnonzero(best < tau)]
    return 1.0 - len(missing) / len(job_vecs), missing


def rerank(
    candidates: list[tuple[int, float]], coverage: dict[int, float], alpha: float = 0.5
) -> list[tuple[int, float]]:
    """score = alpha * (retrieval score / best retrieval score) + (1 - alpha) * coverage."""
    top = max((s for _, s in candidates), default=0.0) or 1.0
    scored = [(d, alpha * s / top + (1 - alpha) * coverage.get(d, 0.0)) for d, s in candidates]
    return sorted(scored, key=lambda x: -x[1])
