"""Torch-free bge-small embeddings (ONNX Runtime on CPU) for the app.

The index was embedded with sentence-transformers on a GPU; this reproduces the same vectors
(CLS pooling + L2 normalize) without PyTorch, which would not fit a 1 GB host next to the
skill model. `scripts/embed_parity.py` checks the two agree.
"""

from __future__ import annotations

import numpy as np

MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction


class OnnxEmbedder:
    def __init__(self, model_id: str = MODEL_ID, max_length: int = 512, batch_size: int = 32,
                 threads: int | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(hf_hub_download(model_id, "onnx/model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length, self.batch_size = max_length, batch_size
        self.dim = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for b in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[b : b + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
            hidden = self.session.run(None, feed)[0]
            cls = hidden[:, 0]
            out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def query(self, text: str) -> np.ndarray:
        return self.encode([QUERY_PREFIX + text])[0]
