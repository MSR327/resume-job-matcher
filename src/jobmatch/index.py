"""The searchable job index: BM25 + dense vectors + extracted skills, saved as plain files.

Built once on a GPU (Kaggle), loaded by the app. Layout of a saved index directory:
    jobs.parquet, vectors.npy (float16), bm25/, skills.json, skill_vecs.npz, manifest.json
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd


def doc_text(jobs: pd.DataFrame) -> list[str]:
    """What gets embedded and BM25-indexed for each job."""
    return (jobs["title"] + "\n" + jobs["description"]).tolist()


@dataclass
class JobIndex:
    jobs: pd.DataFrame
    vectors: np.ndarray  # [n_jobs, dim] float16, L2-normalized
    bm25: bm25s.BM25
    skills: list[list[dict]]  # per job: [{"type": "SKILL"|"KNOWLEDGE", "text": str}]
    skill_vecs: list[np.ndarray]  # per job: [n_skills, dim] float16, L2-normalized
    manifest: dict

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.jobs.to_parquet(p / "jobs.parquet", index=False)
        np.save(p / "vectors.npy", self.vectors)
        self.bm25.save(str(p / "bm25"))
        (p / "skills.json").write_text(json.dumps(self.skills))
        dim = self.vectors.shape[1]
        flat = np.concatenate([v.reshape(-1, dim) for v in self.skill_vecs]) if self.skill_vecs else np.zeros((0, dim))
        offsets = np.cumsum([0] + [len(v) for v in self.skill_vecs])
        np.savez(p / "skill_vecs.npz", flat=flat.astype(np.float16), offsets=offsets)
        (p / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> JobIndex:
        p = Path(path)
        z = np.load(p / "skill_vecs.npz")
        flat, off = z["flat"], z["offsets"]
        return cls(
            jobs=pd.read_parquet(p / "jobs.parquet"),
            vectors=np.load(p / "vectors.npy"),
            bm25=bm25s.BM25.load(str(p / "bm25")),
            skills=json.loads((p / "skills.json").read_text()),
            skill_vecs=[flat[off[i] : off[i + 1]] for i in range(len(off) - 1)],
            manifest=json.loads((p / "manifest.json").read_text()),
        )


def _dedupe(skills: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in skills:
        key = (s["type"], s["text"].strip().lower())
        if key[1] and key not in seen:
            seen.add(key)
            out.append({"type": s["type"], "text": s["text"].strip()})
    return out


def build_index(
    jobs: pd.DataFrame,
    embed_docs: Callable[[list[str]], np.ndarray],
    extract_skills: Callable[[list[str]], list[list[dict]]],
    version: str,
    extra_manifest: dict | None = None,
) -> JobIndex:
    jobs = jobs.reset_index(drop=True)
    texts = doc_text(jobs)
    vectors = np.asarray(embed_docs(texts), dtype=np.float16)
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)

    skills = [_dedupe(s) for s in extract_skills(jobs["description"].tolist())]
    # Embed every distinct skill phrase once, then look up per job.
    phrases = sorted({s["text"] for job in skills for s in job})
    dim = vectors.shape[1]
    pvec = np.asarray(embed_docs(phrases), dtype=np.float16) if phrases else np.zeros((0, dim), np.float16)
    pos = {t: i for i, t in enumerate(phrases)}
    skill_vecs = [pvec[[pos[s["text"]] for s in job]].reshape(-1, dim) for job in skills]

    manifest = {"version": version, "n_jobs": len(jobs), "dim": dim, "created": int(time.time()),
                **(extra_manifest or {})}
    return JobIndex(jobs, vectors, bm25, skills, skill_vecs, manifest)
