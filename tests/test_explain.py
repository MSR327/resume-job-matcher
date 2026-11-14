import json

from jobmatch.explain import build_messages, check_citations, explain, parse_json
from jobmatch.llm import FakeLLM, LLMError

POSTING = (
    "We are hiring an ML Engineer.\nYou will deploy models with   Docker and Kubernetes.\n"
    "Experience with PyTorch is required. Nice to have: Spark."
)
JOB = {"job_id": "7", "title": "ML Engineer", "company": "Acme", "description": POSTING}


def _good(**over):
    obj = {
        "fit_summary": "Strong ML background.",
        "matched_skills": ["PyTorch", "Docker"],
        "missing_skills": ["Spark"],
        "evidence": [
            {"claim": "Needs deployment", "quote": "deploy models with Docker and Kubernetes"},  # whitespace differs
            {"claim": "PyTorch", "quote": "EXPERIENCE WITH PYTORCH IS REQUIRED"},  # case differs
            {"claim": "Made up", "quote": "10 years of Rust required"},  # not in posting
        ],
    }
    obj.update(over)
    return json.dumps(obj)


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_parse_json_rejects_garbage():
    import pytest

    with pytest.raises(ValueError):
        parse_json("no json here")


def test_check_citations_normalizes_whitespace_case_and_quotes():
    ev = [
        {"claim": "a", "quote": "deploy  models with docker"},
        {"claim": "b", "quote": "“Experience with PyTorch”"},  # curly quotes around it
        {"claim": "c", "quote": "Kubernetes and Docker"},  # reordered: not verbatim
        {"claim": "d", "quote": ""},
    ]
    kept = check_citations(ev, POSTING)
    assert [e["claim"] for e in kept] == ["a", "b"]


def test_messages_fence_untrusted_text():
    msgs = build_messages("RESUME TEXT", JOB, ["PyTorch"], ["Spark"])
    user = msgs[-1]["content"]
    assert "<resume>\nRESUME TEXT\n</resume>" in user and "<posting>" in user
    assert "data" in msgs[0]["content"].lower()  # system prompt says fenced text is data, not instructions


def test_explain_drops_hallucinated_quote_and_reports_rate():
    fake = FakeLLM([_good()])
    out = explain(fake, "resume", JOB, matched=["PyTorch"], missing=["Spark"])
    assert out.ok and len(out.evidence) == 2
    assert out.citations_total == 3 and out.citations_kept == 2
    assert out.missing_skills == ["Spark"]


def test_explain_retries_once_then_succeeds():
    fake = FakeLLM(["not json", _good()])
    out = explain(fake, "resume", JOB, matched=[], missing=[])
    assert out.ok and out.attempts == 2


def test_explain_falls_back_after_two_failures():
    fake = FakeLLM([LLMError("429"), "still not json"])
    out = explain(fake, "resume", JOB, matched=["PyTorch"], missing=["Spark"])
    assert not out.ok and out.attempts == 2 and out.error
    # The fallback still carries the deterministic skill overlap so the UI can show something.
    assert out.matched_skills == ["PyTorch"] and out.missing_skills == ["Spark"]


def test_explain_coerces_bad_field_types():
    fake = FakeLLM([_good(matched_skills="PyTorch", evidence=[{"claim": "x"}, "junk"])])
    out = explain(fake, "resume", JOB, matched=[], missing=[])
    assert out.ok and out.matched_skills == ["PyTorch"] and out.evidence == []
