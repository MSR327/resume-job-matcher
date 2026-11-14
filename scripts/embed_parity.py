"""Check the app's ONNX embedder reproduces the sentence-transformers vectors used to build the index.

    python scripts/embed_parity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from jobmatch.embed import MODEL_ID, QUERY_PREFIX, OnnxEmbedder
from jobmatch.index import doc_text


def main() -> None:
    jobs = pd.read_parquet(hf_hub_download("srish327/tech-jobs-index", "v1/jobs.parquet", repo_type="dataset"))
    sample = jobs.sample(50, random_state=0)
    texts = doc_text(sample) + [QUERY_PREFIX + "ML engineer, PyTorch, Docker"] + ["SQL", "stakeholder management"]
    a = SentenceTransformer(MODEL_ID).encode(texts, normalize_embeddings=True)
    b = OnnxEmbedder().encode(texts)
    cos = (a * b).sum(1)
    print(f"cosine ONNX vs sentence-transformers: min {cos.min():.6f}, mean {cos.mean():.6f}")
    stored = np.load(hf_hub_download("srish327/tech-jobs-index", "v1/vectors.npy", repo_type="dataset"))
    cs = (stored[sample.index].astype(np.float32) * b[:50]).sum(1)
    print(f"cosine ONNX vs published index vectors (fp16, built on GPU): min {cs.min():.4f}, mean {cs.mean():.4f}")
    assert cos.min() > 0.999 and cs.min() > 0.99


if __name__ == "__main__":
    main()
