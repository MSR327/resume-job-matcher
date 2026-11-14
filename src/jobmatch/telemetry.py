"""Per-request telemetry as JSON lines: stage latencies, tokens, cost, citation counts.

Numbers only. `add()` refuses strings, so resume or posting text cannot end up in the log by accident.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class _Record:
    def __init__(self):
        self.data: dict[str, float] = {"ts": time.time()}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.data[f"{name}_s"] = self.data.get(f"{name}_s", 0.0) + time.perf_counter() - t0

    def add(self, **values: float):
        for k, v in values.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"telemetry field {k!r} must be a number")
            self.data[k] = self.data.get(k, 0) + v


class Telemetry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def request(self):
        rec = _Record()
        with rec.stage("total"):
            yield rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec.data) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    """p50/p95 per stage, mean tokens and cost, overall citation pass rate."""
    stages = sorted({k for r in rows for k in r if k.endswith("_s")})

    def pct(q):
        return {s: float(np.percentile([r[s] for r in rows if s in r], q)) for s in stages}

    tot = sum(r.get("citations_total", 0) for r in rows)
    return {
        "n": len(rows),
        "latency_p50": pct(50) if rows else {},
        "latency_p95": pct(95) if rows else {},
        "mean_tokens": float(np.mean([r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rows]))
        if rows
        else 0.0,
        "mean_cost_usd": float(np.mean([r.get("cost_usd", 0) for r in rows])) if rows else 0.0,
        "citation_pass_rate": sum(r.get("citations_kept", 0) for r in rows) / tot if tot else float("nan"),
    }
