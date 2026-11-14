"""Blind human labeling for the judge-agreement check (L3).

    streamlit run tools/label_app.py

Shows 50 resume-job pairs (stratified by the judge's grade, which is hidden) and saves your
0-3 grades to data/eval/human_labels.jsonl. You can stop and come back; progress is kept.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from jobmatch.judge import RUBRIC

JUDGED = Path("data/eval/judgments.jsonl")
SYNTH = Path("data/eval/synth.jsonl")
OUT = Path("data/eval/human_labels.jsonl")
N, SEED = 50, 7

st.set_page_config(page_title="Label pairs", page_icon="🏷️", layout="wide")


@st.cache_data
def load():
    jobs = pd.read_parquet(hf_hub_download("srish327/tech-jobs-index", "v1/jobs.parquet", repo_type="dataset"))
    jobs = jobs.set_index("job_id")
    resumes = {json.loads(x)["job_id"]: json.loads(x)["resume"] for x in SYNTH.read_text().splitlines()}
    rows = [json.loads(x) for x in JUDGED.read_text().splitlines()]
    rows = [r for r in rows if r["grade"] is not None]
    rng = random.Random(SEED)
    by_grade = {g: [r for r in rows if r["grade"] == g] for g in range(4)}
    for group in by_grade.values():
        rng.shuffle(group)
    sample, g = [], 0
    while len(sample) < N and any(by_grade.values()):  # round-robin over grades: balanced sample
        if by_grade[g % 4]:
            sample.append(by_grade[g % 4].pop())
        g += 1
    rng.shuffle(sample)
    return jobs, resumes, [(r["resume_job_id"], r["job_id"]) for r in sample]


jobs, resumes, pairs = load()
done = {}
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done[(r["resume_job_id"], r["job_id"])] = r["grade"]
todo = [p for p in pairs if p not in done]

st.title("🏷️ Grade resume ↔ job fit")
st.progress(len(pairs) - len(todo), text=f"{len(pairs) - len(todo)} / {len(pairs)} labeled")
if not todo:
    st.success("All done, thank you! Now run: python scripts/agreement.py")
    st.stop()

rid, jid = todo[0]
job = jobs.loc[jid]
st.markdown("**Rubric**\n\n" + RUBRIC.replace("\n", "  \n"))
left, right = st.columns(2)
left.subheader("Resume")
left.write(resumes[rid])
right.subheader(f"Job: {job['title']}")
right.write(job["description"][:3000] + ("..." if len(job["description"]) > 3000 else ""))
cols = st.columns(4)
for g, label in enumerate(["0 · not relevant", "1 · weak", "2 · good", "3 · excellent"]):
    if cols[g].button(label, width="stretch", key=f"g{g}"):
        with OUT.open("a") as f:
            f.write(json.dumps({"resume_job_id": rid, "job_id": jid, "grade": g}) + "\n")
        st.rerun()
