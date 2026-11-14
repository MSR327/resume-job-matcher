"""Streamlit rendering of match results, separate from model loading so it can be tested without models."""

from __future__ import annotations

import streamlit as st

from .pipeline import Match


def chips(items: list[str], color: str, limit: int = 12) -> str:
    shown = " ".join(f":{color}-badge[{s.replace('[', '(').replace(']', ')')}]" for s in items[:limit])
    return shown + (f" *+{len(items) - limit} more*" if len(items) > limit else "")


def render_match(rank: int, m: Match) -> None:
    j = m.job
    with st.container(border=True):
        left, right = st.columns([4, 1])
        title = f"[{j['title']}]({j['url']})" if j.get("url") else j["title"]
        left.markdown(f"#### {rank}. {title}\n**{j['company']}** · {j.get('location') or 'location n/a'}")
        right.metric("Skill coverage", f"{m.coverage:.0%}",
                     help="Share of the job's extracted skills your resume covers (soft match on skill embeddings)")
        e = m.explanation
        if e is not None and e.ok:
            st.markdown(e.fit_summary)
            have, miss = e.matched_skills, e.missing_skills
        else:
            if e is not None:
                st.info("Explanation unavailable right now (LLM busy or failed), showing skill overlap only.")
            have, miss = m.matched, m.missing
        if have:
            st.markdown("**You have:** " + chips(have, "green"))
        if miss:
            st.markdown("**Missing:** " + chips(miss, "orange"))
        if e is not None and e.ok and e.evidence:
            with st.expander(f"Evidence from the posting ({e.citations_kept}/{e.citations_total} quotes verified)"):
                for ev in e.evidence:
                    st.markdown(f"> {ev['quote']}\n\n{ev['claim']}")
        with st.expander("Job description"):
            st.write(j["description"][:4000] + ("..." if len(j["description"]) > 4000 else ""))


def render_footer(matches: list[Match], elapsed: float, index_version: str) -> None:
    ex = [m.explanation for m in matches if m.explanation]
    kept, total = sum(e.citations_kept for e in ex), sum(e.citations_total for e in ex)
    st.caption(
        f"{elapsed:.1f} s · {sum(e.prompt_tokens + e.completion_tokens for e in ex):,} LLM tokens · "
        f"~${sum(e.cost_usd for e in ex):.4f} at list price · quotes verified {kept}/{total} · index {index_version}"
    )
