"""L1 eval set: sample jobs from the index, have an LLM write a fitting resume for each.

    python scripts/make_synth.py --n 300 [--model qwen/qwen3.8-27b]

Reads GROQ_API_KEY from the environment or a git-ignored .env file. Resumable: rows already in
the output file are skipped, so a run cut off by a daily rate limit continues where it stopped.
Output: data/eval/synth.jsonl (synthetic text only, no real person's data).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from jobmatch.llm import SYNTH_MODEL, groq
from jobmatch.synth import make_resume

OUT = Path("data/eval/synth.jsonl")


def api_key() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    env = Path(".env")
    for line in env.read_text().splitlines() if env.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == "GROQ_API_KEY" and v.strip():
            return v.strip().strip('"')
    raise SystemExit("GROQ_API_KEY not set (env var or .env)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default=SYNTH_MODEL)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = pd.read_parquet(hf_hub_download("srish327/tech-jobs-index", f"{args.version}/jobs.parquet",
                                           repo_type="dataset"))
    order = list(range(len(jobs)))
    random.Random(args.seed).shuffle(order)  # fixed sample: same jobs on every rerun

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["job_id"] for line in OUT.read_text().splitlines()} if OUT.exists() else set()
    llm = groq(api_key(), model=args.model, rate_limit_retries=4, timeout=60)
    rng = random.Random(args.seed)
    kept, rejected, streak, t0 = len(done), 0, 0, time.time()
    with OUT.open("a") as f:
        for i in order:
            if kept >= args.n:
                break
            job = jobs.iloc[i].to_dict()
            if job["job_id"] in done:
                continue
            row = make_resume(llm, job, rng)
            if row is None:
                rejected += 1
                streak += 1
                # Circuit breaker: a run of failures means a broken setup (bad model name, key,
                # quota), not unlucky jobs. Stop instead of silently walking the whole corpus.
                if streak >= 15:
                    raise SystemExit(f"15 failures in a row after {kept} kept: check model/key/quota")
                continue
            streak = 0
            row["model"] = args.model
            f.write(json.dumps(row) + "\n")
            f.flush()
            kept += 1
            if kept % 10 == 0:
                print(f"{kept}/{args.n} kept, {rejected} rejected, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {kept} resumes in {OUT} ({rejected} rejected for copying or errors)")


if __name__ == "__main__":
    main()
