"""Explanation quality + cost/latency, through the exact app pipeline (ONNX models, Groq, citation check).

    python scripts/eval_explain.py [--n 20]

- citation pass rate: share of LLM quotes found verbatim (normalized) in the posting
- missing-skill precision (lexical, approximate): share of "missing" skills whose text does not
  appear in the resume. A missing skill that the resume literally mentions is a clear error.
- latency per stage (p50/p95), tokens and $/request at list price, LLM failure rate
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from skillex.onnx_predict import OnnxSkillExtractor

from jobmatch.config import ALPHA, SKILL_MODEL, TAU
from jobmatch.embed import OnnxEmbedder
from jobmatch.index import JobIndex, clean_skills
from jobmatch.llm import groq, load_api_key
from jobmatch.pipeline import Matcher
from jobmatch.synth import split_dev_test
from jobmatch.telemetry import Telemetry, summarize


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/explanations.json")
    args = ap.parse_args()

    root = snapshot_download("srish327/tech-jobs-index", repo_type="dataset", allow_patterns=["v1/*"])
    idx = JobIndex.load(f"{root}/v1", mmap=True)
    emb, ext = OnnxEmbedder(threads=4), OnnxSkillExtractor(SKILL_MODEL, threads=4, low_memory=True)
    tel = Telemetry(Path(tempfile.mkdtemp()) / "t.jsonl")
    matcher = Matcher(idx, emb.query, emb.encode,
                      lambda t: [s["text"] for s in clean_skills(ext.extract(t)["spans"])],
                      llm=groq(load_api_key(), rate_limit_retries=6, timeout=60), telemetry=tel, alpha=ALPHA, tau=TAU)

    items = [json.loads(x) for x in Path("data/eval/synth.jsonl").read_text().splitlines()]
    _, test = split_dev_test(items, 100)
    queries = test[50:50 + args.n]  # after the CI-gate queries

    miss_total = miss_absent = 0
    failed_quotes = []
    for i, q in enumerate(queries, 1):
        resume = _norm(q["resume"])
        for m in matcher.match(q["resume"], k=5):
            e = m.explanation
            if not (e and e.ok):
                continue
            for s in e.missing_skills:
                miss_total += 1
                miss_absent += f" {_norm(s)} " not in f" {resume} "
            if e.citations_kept < e.citations_total and len(failed_quotes) < 10:
                failed_quotes.append({"job": m.job["title"], "kept": e.citations_kept, "total": e.citations_total})
        print(f"{i}/{len(queries)}", flush=True)

    s = summarize(tel.read())
    rows = tel.read()
    out = {
        "n_resumes": len(queries), "k": 5,
        "citation_pass_rate": s["citation_pass_rate"],
        "missing_skill_precision_lexical": miss_absent / miss_total if miss_total else None,
        "n_missing_skills": miss_total,
        "llm_failure_rate": sum(r.get("llm_failed", 0) for r in rows) / max(1, sum(r.get("n_results", 0) for r in rows)),
        "latency_p50_s": s["latency_p50"], "latency_p95_s": s["latency_p95"],
        "mean_tokens_per_request": s["mean_tokens"], "mean_cost_usd_per_request": s["mean_cost_usd"],
        "examples_with_dropped_quotes": failed_quotes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
