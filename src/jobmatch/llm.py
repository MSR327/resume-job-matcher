"""LLM clients behind one small protocol.

Groq (demo), vLLM (bulk eval on Kaggle) and llama.cpp's llama-server (the distilled student from
explain-distill) all serve the OpenAI chat-completions API, so one client covers them and the eval measures
exactly the code path the demo runs. `FallbackLLM` chains two of them (student first, Groq if it fails). `FakeLLM` replays
canned answers for tests, so CI never touches the network.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"  # explanations + judge
SYNTH_MODEL = "qwen/qwen3.8-27b"  # writes synthetic resumes: a different model family than the judge

# Both are reasoning models; hidden "thinking" tokens cost latency and budget, and these tasks
# (short JSON answers) don't need much of it. Checked against the Groq API on 2026-11-15.
REASONING = {"openai/gpt-oss-120b": "low", "openai/gpt-oss-20b": "low", "qwen/qwen3.8-27b": "none"}

# USD per 1M tokens (input, output), public list prices. Used for the "$ / request" estimate only;
# re-check the provider's pricing page before quoting numbers. Unknown models count as $0
# (e.g. a self-hosted vLLM model on a free Kaggle GPU).
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
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


def load_api_key(name: str = "GROQ_API_KEY", env_file: str = ".env") -> str | None:
    """Key from the environment, else from a git-ignored .env file. Never logged or printed."""
    import os
    from pathlib import Path

    if os.environ.get(name):
        return os.environ[name]
    p = Path(env_file)
    for line in p.read_text().splitlines() if p.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == name and v.strip():
            return v.strip().strip('"')
    return None


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
        extra_body: dict | None = None,
        max_wait: float = 60.0,
    ):
        """`rate_limit_retries`: extra tries after HTTP 429 (0 for the app: fail fast and fall back;
        a few for batch jobs, which can afford to wait)."""
        self.rate_limit_retries, self._sleep = rate_limit_retries, sleep
        self.extra_body = extra_body or {}
        self.max_wait = max_wait  # cap on a single rate-limit wait (short for the interactive app)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, messages, *, temperature=0.0, max_tokens=700, json_mode=False) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                **self.extra_body}
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
            log.info("HTTP 429 from %s, waiting %.0fs (try %d)", self.model, min(wait, self.max_wait), attempt + 1)
            self._sleep(min(wait, self.max_wait))
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
    if model in REASONING:
        kw.setdefault("extra_body", {"reasoning_effort": REASONING[model]})
    return OpenAICompatLLM(GROQ_URL, model, api_key=api_key, **kw)


def vllm(model: str, base_url: str = "http://localhost:8000/v1", **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(base_url, model, **kw)


def llama_server(base_url: str, model: str = "student", api_key: str | None = None, **kw) -> OpenAICompatLLM:
    """A distilled student served by llama.cpp's `llama-server` (OpenAI-compatible, project 3 / explain-distill).

    Qwen3 students were trained on prompts with thinking turned off, so the server must render the chat template
    the same way: `enable_thinking: false` goes through as a template argument.
    """
    kw.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})
    return OpenAICompatLLM(base_url.rstrip("/") + "/v1", model, api_key=api_key, **kw)


class FallbackLLM:
    """Try `primary` (e.g. Groq), and on any LLMError use `fallback` (e.g. the self-hosted distilled student).

    The response's `model` says which one answered, so telemetry and cost reflect the path actually taken.
    """

    def __init__(self, primary: LLM, fallback: LLM):
        self.primary, self.fallback = primary, fallback
        self.model = getattr(primary, "model", "primary")

    def complete(self, messages, *, temperature=0.0, max_tokens=700, json_mode=False) -> LLMResponse:
        try:
            return self.primary.complete(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        except LLMError as e:
            log.info("primary LLM failed (%s); using fallback", e)
            return self.fallback.complete(messages, temperature=temperature, max_tokens=max_tokens,
                                          json_mode=json_mode)


class FakeLLM:
    """Replays scripted replies. A reply that is an exception gets raised instead.

    `replies` is a list (used in order) or a function messages -> reply, which keeps tests
    deterministic when calls run in parallel threads. Thread-safe.
    """

    def __init__(self, replies, model: str = "fake"):
        self.replies = replies if callable(replies) else list(replies)
        self.model = model
        self.calls: list[list[dict]] = []
        self._lock = threading.Lock()

    def complete(self, messages, *, temperature=0.0, max_tokens=700, json_mode=False) -> LLMResponse:
        with self._lock:
            self.calls.append(messages)
            reply = self.replies(messages) if callable(self.replies) else self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply, prompt_tokens=100, completion_tokens=50, model=self.model)
