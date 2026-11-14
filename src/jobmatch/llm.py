"""LLM clients behind one small protocol.

Groq (demo) and vLLM (bulk eval on Kaggle) both serve the OpenAI chat-completions API, so one
client covers both and the eval measures exactly the code path the demo runs. `FakeLLM` replays
canned answers for tests, so CI never touches the network.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# USD per 1M tokens (input, output), public list prices. Used for the "$ / request" estimate only;
# re-check the provider's pricing page before quoting numbers. Unknown models count as $0
# (e.g. a self-hosted vLLM model on a free Kaggle GPU).
PRICES = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


class LLMError(RuntimeError):
    """Any failure the caller should treat as "no answer": HTTP error, timeout, empty reply."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""


class LLM(Protocol):
    def complete(
        self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 700, json_mode: bool = False
    ) -> LLMResponse: ...


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


class OpenAICompatLLM:
    """Chat-completions client for any OpenAI-compatible server (Groq, vLLM, ...)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        rate_limit_retries: int = 0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """`rate_limit_retries`: extra tries after HTTP 429 (0 for the app: fail fast and fall back;
        a few for batch jobs, which can afford to wait)."""
        self.rate_limit_retries, self._sleep = rate_limit_retries, sleep
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, messages, *, temperature=0.0, max_tokens=700, json_mode=False) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        for attempt in range(self.rate_limit_retries + 1):
            try:
                r = self._client.post(self.url, json=body, headers=self._headers)
            except httpx.HTTPError as e:  # timeouts, connection errors
                raise LLMError(f"{type(e).__name__} calling {self.model}") from None
            if r.status_code != 429 or attempt == self.rate_limit_retries:
                break
            # Free tiers rate-limit per minute: wait as told (capped), then try again.
            try:
                wait = float(r.headers.get("retry-after", 2 ** (attempt + 2)))
            except ValueError:
                wait = 2.0 ** (attempt + 2)
            self._sleep(min(wait, 60.0))
        if r.status_code != 200:
            # Status only: response bodies can echo request details, and we never log keys or resumes.
            raise LLMError(f"HTTP {r.status_code} from {self.model}")
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError(f"malformed response from {self.model}") from None
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.perf_counter() - t0,
            model=self.model,
        )


def groq(api_key: str, model: str = GROQ_MODEL, **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(GROQ_URL, model, api_key=api_key, **kw)


def vllm(model: str, base_url: str = "http://localhost:8000/v1", **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(base_url, model, **kw)


class FakeLLM:
    """Replays scripted replies in order. A reply that is an exception gets raised instead."""

    def __init__(self, replies: list, model: str = "fake"):
        self.replies = list(replies)
        self.model = model
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, temperature=0.0, max_tokens=700, json_mode=False) -> LLMResponse:
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, prompt_tokens=100, completion_tokens=50, model=self.model)
