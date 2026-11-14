"""Stats: latency per stage, tokens, cost and citation pass rate from this app's telemetry (numbers only)."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from jobmatch.telemetry import Telemetry, summarize

st.set_page_config(page_title="Stats · Resume → Job Matcher", page_icon="📈", layout="wide")
st.title("📈 Stats")
st.caption("Per-request telemetry from this app instance: timings, token counts, estimated cost, quote checks. "
           "No resume text is logged. Resets when the app restarts.")

rows = Telemetry(Path(tempfile.gettempdir()) / "jobmatch_telemetry.jsonl").read()
if not rows:
    st.info("No requests yet. Run a match on the main page.")
    st.stop()

s = summarize(rows)
c = st.columns(4)
c[0].metric("Requests", s["n"])
c[1].metric("p50 total latency", f"{s['latency_p50'].get('total_s', 0):.2f} s")
c[2].metric("Mean cost / request", f"${s['mean_cost_usd']:.4f}")
rate = s["citation_pass_rate"]
c[3].metric("Quotes verified", "n/a" if math.isnan(rate) else f"{rate:.0%}")

stages = [k for k in s["latency_p50"] if k != "total_s"]
lat = pd.DataFrame({"p50 (s)": [s["latency_p50"][k] for k in stages],
                    "p95 (s)": [s["latency_p95"][k] for k in stages]}, index=[k[:-2] for k in stages])
st.subheader("Latency by stage")
st.bar_chart(lat, horizontal=True, stack=False)
st.dataframe(lat.style.format("{:.3f}"), width="stretch")

df = pd.DataFrame(rows)
st.subheader("LLM")
cols = [c for c in ["prompt_tokens", "completion_tokens", "cost_usd", "llm_ok", "llm_failed",
                    "citations_total", "citations_kept"] if c in df]
st.dataframe(df[cols].describe().T.style.format("{:.3f}"), width="stretch")
