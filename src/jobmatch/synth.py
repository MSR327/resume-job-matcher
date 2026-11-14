"""Synthetic eval set (L1): for a job, an LLM writes a resume of someone who fits it.

The eval then asks: is the source job in the top 10 for that resume? To keep this from being a
string-matching test, the prompt asks for paraphrase and realistic gaps, and resumes that copy
the posting (4-gram overlap above a threshold) are rejected and reported.
"""

from __future__ import annotations

import random

from .explain import parse_json
from .llm import LLM, LLMError
from .metrics import ngram_overlap

SYSTEM = """You write realistic, anonymous resumes for evaluation data. Output one JSON object:
{"resume": "plain-text resume, 120-220 words"}
The person is a plausible fit for the job but not a perfect one: they have most core skills,
lack 1-2 nice-to-haves, and have some unrelated experience. Use your own wording: do not copy
phrases from the posting, describe experience in terms of projects and results.
No real names, emails, phone numbers or company names: use "Company A", "University B"."""

SENIORITY = ["junior (0-2 years)", "mid-level (3-5 years)", "senior (6+ years)"]


def synth_messages(job: dict, seniority: str) -> list[dict]:
    user = (f"Seniority: {seniority}\n\n<posting>\nTitle: {job['title']}\n\n"
            f"{job['description'][:4000]}\n</posting>")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def make_resume(llm: LLM, job: dict, rng: random.Random, max_overlap: float = 0.10, tries: int = 2) -> dict | None:
    """One synthetic resume for `job`, or None if every try failed or copied too much."""
    seniority = rng.choice(SENIORITY)
    for _ in range(tries):
        try:
            text = str(parse_json(llm.complete(synth_messages(job, seniority), temperature=0.8,
                                               json_mode=True).text).get("resume", "")).strip()
        except (LLMError, ValueError):
            continue
        if len(text.split()) < 60:
            continue
        ov = ngram_overlap(text, job["description"], n=4)
        if ov <= max_overlap:
            return {"job_id": job["job_id"], "seniority": seniority, "resume": text, "overlap4": ov}
    return None


def split_dev_test(items: list[dict], n_dev: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    items = sorted(items, key=lambda x: x["job_id"])
    random.Random(seed).shuffle(items)
    return items[:n_dev], items[n_dev:]
