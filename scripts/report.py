"""Render results/*.json as Markdown for the README (numbers come from files, never typed by hand).

    python scripts/report.py > results/RESULTS.md
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("results")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    out = []
    l1 = load("l1_retrieval.json")
    if l1:
        out += [(f"**L1: synthetic resumes** (test n = {l1['n_test']}; tuned on dev n = {l1['n_dev']}; "
                f"mean 4-gram overlap with the source posting {l1['overlap4_mean']:.1%})"), "",
                "| configuration | Recall@1 | Recall@5 | Recall@10 | MRR |", "|---|---|---|---|---|"]
        for name, m in l1["test"].items():
            tag = " ← app" if name == l1["dev_choice"] else ""
            out.append(f"| {name}{tag} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['MRR']:.3f} |")
        out += ["", "Paired bootstrap (95% CI, test):", ""]
        for k, (d, lo, hi) in l1["ci95"].items():
            out.append(f"- {k}: **{d:+.3f}** ({lo:+.3f} to {hi:+.3f})")
        tuned = ", ".join(f"{b}: τ={v['tau']}, α={v['alpha']}" for b, v in l1["tuned"].items())
        out += ["", f"Tuned on dev: {tuned}. App configuration (best dev MRR): `{l1['dev_choice']}`.", ""]
    l2 = load("l2_judge.json")
    if l2:
        out += [f"**L2: LLM judge** ({l2['judge']}, {l2['n_resumes']} test resumes, {l2['n_pairs']} judged pairs)", "",
                "| configuration | nDCG@10 | P@10 (grade ≥ 2) |", "|---|---|---|"]
        for name, m in l2["configs"].items():
            out.append(f"| {name} | {m['nDCG@10']:.3f} | {m['P@10(grade>=2)']:.3f} |")
        out += ["", (f"Sanity check: the judge gives each resume's source job a mean grade of "
                f"{l2['source_job_grade_mean']:.2f} / 3 (n = {l2['n_source_judged']}). "
                f"Grade distribution: {l2['grade_distribution']}."), ""]
    l3 = load("l3_agreement.json")
    if l3:
        out += [f"**L3: judge vs. human** (n = {l3['n']} pairs, labeled blind)", "",
                (f"- quadratic-weighted κ **{l3['weighted_kappa_quadratic']:.2f}**, exact agreement "
                f"{l3['exact_agreement']:.0%}, within one grade {l3['within_one']:.0%}"),
                (f"- relevant / not relevant (grade ≥ 2) agreement {l3['relevant_agreement(grade>=2)']:.0%}; "
                f"judge minus human {l3['judge_minus_human_mean']:+.2f} grades on average"), ""]
    ex = load("explanations.json")
    if ex:
        p50, p95 = ex["latency_p50_s"], ex["latency_p95_s"]
        out += [f"**Explanations and serving** ({ex['n_resumes']} test resumes × top {ex['k']}, app pipeline, laptop CPU)", "",
                f"- citation pass rate **{ex['citation_pass_rate']:.0%}** (quotes found verbatim in the posting)",
                (f"- missing-skill precision (lexical) **{ex['missing_skill_precision_lexical']:.0%}** "
                f"(n = {ex['n_missing_skills']})"),
                f"- LLM failure rate {ex['llm_failure_rate']:.1%} (falls back to skill overlap)",
                ("- latency p50: skill extraction {:.2f} s, retrieval {:.3f} s, re-rank {:.2f} s; LLM explanations "
                 "{:.0f} s (p95 {:.0f} s), almost all of it waiting on Groq's free-tier limit of 8k tokens/min, since one "
                 "request needs ~{:.0f}k tokens. Unthrottled, a full request took ~3 s.").format(
                    p50.get("skills_s", 0), p50.get("retrieve_s", 0), p50.get("rerank_s", 0), p50.get("explain_s", 0),
                    p95.get("explain_s", 0), ex["mean_tokens_per_request"] / 1000),
                (f"- {ex['mean_tokens_per_request']:.0f} tokens and ${ex['mean_cost_usd_per_request']:.4f} per request "
                "at list price"), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
