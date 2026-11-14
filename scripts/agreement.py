"""L3: how well does the LLM judge agree with a human? (data/eval/human_labels.jsonl vs judgments.jsonl)

    python scripts/agreement.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jobmatch.metrics import weighted_kappa


def main() -> None:
    judge = {}
    for line in Path("data/eval/judgments.jsonl").read_text().splitlines():
        r = json.loads(line)
        judge[(r["resume_job_id"], r["job_id"])] = r["grade"]
    human = [json.loads(x) for x in Path("data/eval/human_labels.jsonl").read_text().splitlines()]
    h = np.array([r["grade"] for r in human])
    j = np.array([judge[(r["resume_job_id"], r["job_id"])] for r in human])
    conf = np.zeros((4, 4), int)
    for a, b in zip(h, j):
        conf[a, b] += 1
    out = {
        "n": len(h),
        "weighted_kappa_quadratic": weighted_kappa(h.tolist(), j.tolist()),
        "exact_agreement": float((h == j).mean()),
        "within_one": float((abs(h - j) <= 1).mean()),
        "mean_abs_diff": float(abs(h - j).mean()),
        "judge_minus_human_mean": float((j - h).mean()),
        "relevant_agreement(grade>=2)": float(((h >= 2) == (j >= 2)).mean()),
        "confusion_human_rows_judge_cols": conf.tolist(),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/l3_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
