"""UI rendering with fake matches: explained, fallback, and no-LLM paths (no models needed)."""

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def _app():
    from jobmatch.explain import Explanation
    from jobmatch.pipeline import Match
    from jobmatch.ui import render_footer, render_match

    job = {"title": "ML Engineer", "company": "Acme", "location": "", "url": "https://x.test/1",
           "description": "We need PyTorch and Docker."}
    ok = Explanation(ok=True, fit_summary="Strong fit.", matched_skills=["PyTorch"], missing_skills=["Spark"],
                     evidence=[{"claim": "needs PyTorch", "quote": "We need PyTorch"}], citations_total=2,
                     citations_kept=1, prompt_tokens=100, completion_tokens=50, cost_usd=0.0001)
    failed = Explanation(ok=False, error="HTTP 429")
    render_match(1, Match(0, job, 0.9, 0.5, ["PyTorch"], ["Docker"], ok))
    render_match(2, Match(1, job, 0.8, 0.25, ["Docker"], ["Spark", "[weird] skill"], failed))
    render_match(3, Match(2, job, 0.7, 0.0, [], ["Go"], None))
    render_footer([], 1.0, "v1")


def test_render_paths():
    at = st_testing.AppTest.from_function(_app).run()
    assert not at.exception
    md = [m.value for m in at.markdown]
    assert any(v.startswith("#### 1. [ML Engineer](https://x.test/1)") for v in md)
    assert "Strong fit." in md
    assert any("Missing:** :orange-badge[Spark]" in v for v in md)  # LLM's list, not the matcher's
    assert len(at.info) == 1  # only the failed explanation shows the fallback notice
    assert any(":orange-badge[(weird) skill]" in v for v in md)  # brackets can't break badge markup
    assert [m.value for m in at.metric] == ["50%", "25%", "0%"]
