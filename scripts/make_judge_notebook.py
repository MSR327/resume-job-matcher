"""Generate notebooks/02_judge.ipynb pinned to a commit of this repo (the commit holding the pairs file).

    python scripts/make_judge_notebook.py <commit> [test|dev]
"""

import json
import sys


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


def build(commit: str, split: str = "test") -> dict:
    sfx = "" if split == "test" else f"_{split}"
    cells = [
md("""# L2 judge: grade resume-job pairs with an open LLM

IBM Granite-3.3-8B-Instruct (Apache-2.0) grades each (synthetic resume, job) pair 0-3 with the same rubric and prompt
code as `jobmatch.judge`, for nDCG@10 of each retrieval configuration. Groq's free tier (200k tokens/day) could not
grade ~1k pairs, so this runs on a free Kaggle GPU.

**Needs:** GPU T4 x2, Internet on, secret `hg` (HF write token) attached."""),
code(f"""%pip install -q "jinja2>=3.1.6" "jobmatch @ git+https://github.com/MSR327/resume-job-matcher.git@{commit}"
COMMIT = "{commit}"
SPLIT = "{split}"  # test: scored once; dev: used to choose the app configuration
SFX = "{sfx}"
JUDGE = "ibm-granite/granite-3.3-8b-instruct\""""),
code("""# Fail fast: HF token + write access before loading an 8B model.
import os

from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hg")
api = HfApi()
REPO = f"{api.whoami()['name']}/tech-jobs-index"
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
print("HF auth ok ->", REPO)"""),
code("""import json
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download

raw = f"https://raw.githubusercontent.com/MSR327/resume-job-matcher/{COMMIT}/data/eval"
pairs = json.loads(urllib.request.urlopen(f"{raw}/l2_pairs{SFX}.json").read())
resumes = {json.loads(x)["job_id"]: json.loads(x)["resume"]
           for x in urllib.request.urlopen(f"{raw}/synth.jsonl").read().decode().splitlines()}
jobs = pd.read_parquet(hf_hub_download(REPO, "v1/jobs.parquet", repo_type="dataset")).set_index("job_id")
todo = sorted({(q["resume_job_id"], j) for q in pairs["queries"] for js in q["top10"].values() for j in js})
print(len(pairs["queries"]), "resumes,", len(todo), "unique pairs")"""),
code("""import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jobmatch.explain import parse_json
from jobmatch.judge import judge_messages

tok = AutoTokenizer.from_pretrained(JUDGE, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.float16, device_map="auto").eval()
print(torch.cuda.device_count(), "GPUs;", {d: f"{torch.cuda.memory_allocated(d) / 1e9:.1f} GB" for d in range(torch.cuda.device_count())})

prompts = [tok.apply_chat_template(judge_messages(resumes[r], {"title": jobs.loc[j, "title"],
                                                               "description": jobs.loc[j, "description"]}),
                                   tokenize=False, add_generation_prompt=True) for r, j in todo]
order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch: less padding
texts, t0, BS = [None] * len(prompts), time.time(), 16
for b in range(0, len(order), BS):
    idx = order[b:b + BS]
    enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
    for i, seq in zip(idx, out[:, enc["input_ids"].shape[1]:]):
        texts[i] = tok.decode(seq, skip_special_tokens=True)
    if (b // BS) % 10 == 0:
        print(f"judged {min(b + BS, len(order))}/{len(order)}  {(time.time() - t0) / 60:.1f} min", flush=True)"""),
code("""import re


def grade_of(text):
    try:
        g = int(parse_json(text)["grade"])
    except (ValueError, KeyError, TypeError):
        m = re.search(r'"grade"\\s*:\\s*"?([0-3])', text)
        g = int(m.group(1)) if m else None
    return g if g in (0, 1, 2, 3) else None


rows = [{"resume_job_id": r, "job_id": j, "grade": grade_of(t), "is_source": r == j, "model": JUDGE, "raw": t[:300]}
        for (r, j), t in zip(todo, texts)]
os.makedirs("out", exist_ok=True)
with open("out/judgments.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\\n" for r in rows)
grades = [r["grade"] for r in rows]
print("grades:", {g: grades.count(g) for g in (0, 1, 2, 3, None)})
src = [r["grade"] for r in rows if r["is_source"] and r["grade"] is not None]
print("source-job mean grade:", round(sum(src) / max(1, len(src)), 2), "n =", len(src))
for r in rows[:3]:
    print(r["raw"][:200])"""),
code("""api.upload_file(path_or_fileobj="out/judgments.jsonl", path_in_repo=f"eval/judgments{SFX}_granite-3.3-8b.jsonl",
                repo_id=REPO, repo_type="dataset",
                commit_message=f"L2 judgments, {SPLIT} ({len(rows)} pairs, eval commit {COMMIT})")
print("uploaded", f"https://huggingface.co/datasets/{REPO}/blob/main/eval/judgments{SFX}_granite-3.3-8b.jsonl")"""),
]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    with open("notebooks/02_judge.ipynb", "w") as f:
        json.dump(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "test"), f, indent=1)
    print("written for commit", sys.argv[1], *sys.argv[2:])
