"""Resume in, ranked + explained jobs out. Shared by the Streamlit app and the offline eval.

Models are injected as plain callables (embed, extract skills, LLM), so the same pipeline runs with
ONNX on CPU in the app, with PyTorch on Kaggle, and with tiny fakes in tests.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .explain import Explanation, explain
from .index import JobIndex
from .llm import LLM
from .rerank import rerank, skill_coverage
from .retrieve import HybridRetriever
from .telemetry import Telemetry


@dataclass
class Match:
    job_idx: int
    job: dict
    score: float
    coverage: float
    matched: list[str]  # job skills the resume covers (soft match)
    missing: list[str]  # job skills the resume does not cover
    explanation: Explanation | None = None


class Matcher:
    def __init__(
        self,
        index: JobIndex,
        embed_query: Callable[[str], np.ndarray],
        embed_phrases: Callable[[list[str]], np.ndarray],
        extract_skills: Callable[[str], list[str]],
        llm: LLM | None = None,
        telemetry: Telemetry | None = None,
        alpha: float = 0.5,
        tau: float = 0.75,
    ):
        self.index, self.embed_phrases, self.extract_skills = index, embed_phrases, extract_skills
        self.retriever = HybridRetriever(index.bm25, index.vectors, embed_query)
        self.llm, self.telemetry, self.alpha, self.tau = llm, telemetry, alpha, tau

    def match(
        self, resume: str, k: int = 5, mode: str = "hybrid", skill_rerank: bool = True, explain_top: bool = True,
        depth: int = 100, pool: int = 50,
    ) -> list[Match]:
        ctx = self.telemetry.request() if self.telemetry else nullcontext(None)
        with ctx as rec:
            stage = rec.stage if rec else (lambda name: nullcontext())
            with stage("skills"):
                skills = sorted({s.strip() for s in self.extract_skills(resume) if s.strip()}, key=str.lower)
                dim = self.index.vectors.shape[1]
                rvec = np.asarray(self.embed_phrases(skills)) if skills else np.zeros((0, dim), np.float32)
            with stage("retrieve"):
                cands = self.retriever.search(resume, k=pool, depth=depth, mode=mode)
            with stage("rerank"):
                cov = {d: skill_coverage(rvec, self.index.job_skill_vecs(d), self.tau) for d, _ in cands}
                ranked = rerank(cands, {d: c for d, (c, _) in cov.items()}, self.alpha) if skill_rerank else cands
            out = []
            for d, score in ranked[:k]:
                names = [s["text"] for s in self.index.skills[d]]
                miss = set(cov[d][1])
                out.append(Match(d, self.index.jobs.iloc[d].to_dict(), float(score), cov[d][0],
                                 [n for i, n in enumerate(names) if i not in miss], [names[i] for i in sorted(miss)]))
            if self.llm and explain_top:
                with stage("explain"):
                    for m in out:
                        m.explanation = explain(self.llm, resume, m.job, m.matched, m.missing)
            if rec:
                ex = [m.explanation for m in out if m.explanation]
                rec.add(n_resume_skills=len(skills), n_results=len(out),
                        llm_ok=sum(e.ok for e in ex), llm_failed=sum(not e.ok for e in ex),
                        prompt_tokens=sum(e.prompt_tokens for e in ex),
                        completion_tokens=sum(e.completion_tokens for e in ex),
                        cost_usd=sum(e.cost_usd for e in ex),
                        citations_total=sum(e.citations_total for e in ex),
                        citations_kept=sum(e.citations_kept for e in ex))
        return out
