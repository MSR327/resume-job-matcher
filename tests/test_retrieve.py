import bm25s
import numpy as np

from jobmatch.retrieve import HybridRetriever, rrf


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert [d for d, _ in fused[:2]] in ([1, 2], [2, 1])
    assert dict(fused)[3] < dict(fused)[1]
    assert {d for d, _ in fused} == {1, 2, 3, 4}


DOCS = ["python django rest api backend", "react typescript frontend ui", "kubernetes terraform aws devops"]


def _retriever():
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(DOCS, stopwords="en", show_progress=False), show_progress=False)
    vecs = np.eye(3, dtype=np.float32)  # doc i <-> axis i

    def embed(q):  # toy "semantic" embedder
        return np.array([("backend" in q) + 0.0, ("ui" in q) + 0.0, ("cloud" in q) + 0.0], np.float32)

    return HybridRetriever(bm, vecs, embed)


def test_modes():
    r = _retriever()
    assert r.search("terraform", k=1, mode="bm25")[0][0] == 2
    assert r.search("cloud infra", k=1, mode="dense")[0][0] == 2  # no word overlap; dense finds it
    assert r.search("python cloud", k=2, mode="hybrid")[0][0] in (0, 2)
