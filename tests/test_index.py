import numpy as np
import pandas as pd

from jobmatch.index import JobIndex, build_index, clean_skills
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
    assert idx.job_skill_vecs(0).shape == (3, DIM) and idx.vectors.dtype == np.float16
    assert idx.manifest["n_jobs"] == 3 and idx.manifest["version"] == "test"

    idx.save(tmp_path / "v")
    back = JobIndex.load(tmp_path / "v")
    pd.testing.assert_frame_equal(back.jobs, idx.jobs)
    assert np.array_equal(back.vectors, idx.vectors)
    assert back.skills == idx.skills
    assert back.phrases == idx.phrases and np.array_equal(back.phrase_vecs, idx.phrase_vecs)
    assert all(np.array_equal(back.job_skill_vecs(d), idx.job_skill_vecs(d)) for d in range(3))
    assert back.manifest == idx.manifest

    r = HybridRetriever(back.bm25, back.vectors, lambda q: _embed([q])[0])
    assert r.search("terraform kubernetes", k=1, mode="bm25")[0][0] == 2


def test_job_without_skills(tmp_path):
    idx = build_index(_jobs(), _embed, lambda descs: [[] for _ in descs], version="t")
    idx.save(tmp_path / "v")
    back = JobIndex.load(tmp_path / "v")
    assert all(back.job_skill_vecs(d).shape == (0, DIM) for d in range(3))


def test_clean_skills_drops_noise_and_dupes():
    raw = [
        {"type": "SKILL", "text": "skilled"},  # one-word SKILL: filler
        {"type": "KNOWLEDGE", "text": " Python "},  # one-word KNOWLEDGE: keep
        {"type": "KNOWLEDGE", "text": "python"},  # case duplicate
        {"type": "SKILL", "text": "mentor junior engineers"},
        {"type": "SKILL", "text": " ".join(["word"] * 11)},  # a sentence, not a skill
    ]
    assert clean_skills(raw) == [{"type": "KNOWLEDGE", "text": "Python"},
                                 {"type": "SKILL", "text": "mentor junior engineers"}]


def test_shared_phrase_stored_once():
    jobs = _jobs()
    idx = build_index(jobs, _embed, lambda ds: [[{"type": "KNOWLEDGE", "text": "AWS"}] for _ in ds], version="t")
    assert idx.phrases == ["AWS"] and idx.phrase_vecs.shape == (1, DIM)
    assert [list(i) for i in idx.skill_ids] == [[0], [0], [0]]
    assert idx.manifest["n_phrases"] == 1 and idx.manifest["n_spans_kept"] == 3
