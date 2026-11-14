"""The searchable job index: BM25 + dense vectors + extracted skills, saved as plain files.

Built once on a GPU (Kaggle), loaded by the app. Layout of a saved index directory:
    jobs.parquet, vectors.npy (float16), bm25/, skills.json, phrases.json,
    phrase_vecs.npy (float16), skill_ids.npz, manifest.json

Skill phrases repeat a lot across jobs ("Python" appears in thousands), so each distinct phrase
is embedded and stored once (`phrase_vecs`); a job keeps only the row ids of its phrases.
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

MAX_SPAN_WORDS = 10


def doc_text(jobs: pd.DataFrame) -> list[str]:
    """What gets embedded and BM25-indexed for each job."""
    return (jobs["title"] + "\n" + jobs["description"]).tolist()


def keep_span(s: dict) -> bool:
    """Drop spans that are noise for matching.

    One-word SKILL spans are mostly filler adjectives ("skilled", "motivated", "experienced");
    one-word KNOWLEDGE spans ("Python", "AWS") are the most useful ones, so they stay.
    Very long spans are whole sentences, not skills.
    """
    n = len(s["text"].split())
    return (n >= 2 if s["type"] == "SKILL" else n >= 1) and n <= MAX_SPAN_WORDS


@dataclass
class JobIndex:
    jobs: pd.DataFrame
    vectors: np.ndarray  # [n_jobs, dim] float16, L2-normalized
    bm25: bm25s.BM25
    skills: list[list[dict]]  # per job: [{"type": "SKILL"|"KNOWLEDGE", "text": str}]
    phrases: list[str]  # distinct skill phrases
    phrase_vecs: np.ndarray  # [n_phrases, dim] float16, L2-normalized
    skill_ids: list[np.ndarray]  # per job: row ids into phrases/phrase_vecs, aligned with skills
    manifest: dict

    def job_skill_vecs(self, d: int) -> np.ndarray:
        return self.phrase_vecs[self.skill_ids[d]]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.jobs.to_parquet(p / "jobs.parquet", index=False)
        np.save(p / "vectors.npy", self.vectors)
        self.bm25.save(str(p / "bm25"))
        (p / "skills.json").write_text(json.dumps(self.skills))
        (p / "phrases.json").write_text(json.dumps(self.phrases))
        np.save(p / "phrase_vecs.npy", self.phrase_vecs.astype(np.float16))
        flat = np.concatenate(self.skill_ids) if self.skill_ids else np.zeros(0)
        offsets = np.cumsum([0] + [len(v) for v in self.skill_ids])
        np.savez(p / "skill_ids.npz", flat=flat.astype(np.int32), offsets=offsets)
        (p / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> JobIndex:
        p = Path(path)
        z = np.load(p / "skill_ids.npz")
        flat, off = z["flat"], z["offsets"]
        return cls(
            jobs=pd.read_parquet(p / "jobs.parquet"),
            vectors=np.load(p / "vectors.npy"),
            bm25=bm25s.BM25.load(str(p / "bm25")),
            skills=json.loads((p / "skills.json").read_text()),
            phrases=json.loads((p / "phrases.json").read_text()),
            phrase_vecs=np.load(p / "phrase_vecs.npy"),
            skill_ids=[flat[off[i] : off[i + 1]] for i in range(len(off) - 1)],
            manifest=json.loads((p / "manifest.json").read_text()),
        )


def clean_skills(skills: list[dict]) -> list[dict]:
    """Filter noise spans, strip, and dedupe case-insensitively (first spelling wins)."""
    seen, out = set(), []
    for s in skills:
        text = s["text"].strip()
        key = (s["type"], text.lower())
        if text and key not in seen and keep_span({"type": s["type"], "text": text}):
            seen.add(key)
            out.append({"type": s["type"], "text": text})
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

    raw = extract_skills(jobs["description"].tolist())
    skills = [clean_skills(s) for s in raw]
    phrases = sorted({s["text"] for job in skills for s in job})
    dim = vectors.shape[1]
    pvec = np.asarray(embed_docs(phrases), dtype=np.float16) if phrases else np.zeros((0, dim), np.float16)
    pos = {t: i for i, t in enumerate(phrases)}
    skill_ids = [np.array([pos[s["text"]] for s in job], dtype=np.int32) for job in skills]

    manifest = {
        "version": version, "n_jobs": len(jobs), "dim": dim, "created": int(time.time()),
        "n_spans_raw": sum(map(len, raw)), "n_spans_kept": sum(map(len, skills)), "n_phrases": len(phrases),
        **(extra_manifest or {}),
    }
    return JobIndex(jobs, vectors, bm25, skills, phrases, pvec, skill_ids, manifest)
