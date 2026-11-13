import numpy as np
import pandas as pd

from jobmatch.index import JobIndex, build_index
from jobmatch.retrieve import HybridRetriever

DIM = 384


def _jobs():
    return pd.DataFrame({
        "job_id": ["a", "b", "c"],
        "title": ["Backend Engineer", "Frontend Developer", "DevOps Engineer"],
        "company": ["X", "Y", "Z"], "location": ["", "", ""],
        "description": ["Python Django APIs.", "React and TypeScript UIs.", "Kubernetes and Terraform on AWS."],
        "url": ["", "", ""], "listed_at": [0, 0, 0],
    })


def _embed(texts):
    rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
    v = rng.normal(size=(len(texts), DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _skills(descs):
    out = []
    for d in descs:
        words = [w.strip(".") for w in d.split() if w[0].isupper()]
        out.append([{"type": "KNOWLEDGE", "text": w} for w in words])
    out[0].append({"type": "KNOWLEDGE", "text": "python"})  # case-duplicate of "Python" -> deduped
    return out


def test_build_save_load_roundtrip(tmp_path):
    idx = build_index(_jobs(), _embed, _skills, version="test")
    assert [s["text"] for s in idx.skills[0]] == ["Python", "Django", "APIs"]
    assert idx.skill_vecs[0].shape == (3, DIM) and idx.vectors.dtype == np.float16
    assert idx.manifest["n_jobs"] == 3 and idx.manifest["version"] == "test"

    idx.save(tmp_path / "v")
    back = JobIndex.load(tmp_path / "v")
    pd.testing.assert_frame_equal(back.jobs, idx.jobs)
    assert np.array_equal(back.vectors, idx.vectors)
    assert back.skills == idx.skills
    assert all(np.array_equal(a, b) for a, b in zip(back.skill_vecs, idx.skill_vecs))
    assert back.manifest == idx.manifest

    r = HybridRetriever(back.bm25, back.vectors, lambda q: _embed([q])[0])
    assert r.search("terraform kubernetes", k=1, mode="bm25")[0][0] == 2


def test_job_without_skills(tmp_path):
    idx = build_index(_jobs(), _embed, lambda descs: [[] for _ in descs], version="t")
    idx.save(tmp_path / "v")
    back = JobIndex.load(tmp_path / "v")
    assert all(v.shape == (0, DIM) for v in back.skill_vecs)
