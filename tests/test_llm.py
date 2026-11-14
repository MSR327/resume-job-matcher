import json

import httpx
import pytest

from jobmatch.llm import FakeLLM, LLMError, OpenAICompatLLM, estimate_cost, groq


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_parses_text_and_usage():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    llm = OpenAICompatLLM("https://x.test/v1", "m-1", api_key="k", client=_client(handler))
    r = llm.complete([{"role": "user", "content": "hi"}], json_mode=True)
    assert r.text == '{"ok": true}' and r.prompt_tokens == 12 and r.completion_tokens == 5
    assert seen["url"] == "https://x.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "m-1" and seen["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("status", [429, 500])
def test_openai_compat_http_errors_raise_llmerror(status):
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=_client(lambda r: httpx.Response(status)))
    with pytest.raises(LLMError):
        llm.complete([{"role": "user", "content": "hi"}])


def test_api_key_never_in_error_message():
    llm = OpenAICompatLLM("https://x.test/v1", "m", api_key="secret-123",
                          client=_client(lambda r: httpx.Response(401, text="bad key secret-123")))
    with pytest.raises(LLMError) as e:
        llm.complete([{"role": "user", "content": "hi"}])
    assert "secret-123" not in str(e.value)


def test_fake_llm_replays_and_records():
    fake = FakeLLM(["a", LLMError("boom"), "b"])
    assert fake.complete([{"role": "user", "content": "1"}]).text == "a"
    with pytest.raises(LLMError):
        fake.complete([])
    assert fake.complete([]).text == "b"
    assert len(fake.calls) == 3


def test_estimate_cost_known_and_unknown_model():
    assert estimate_cost("openai/gpt-oss-120b", 1_000_000, 0) > 0
    assert estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_rate_limit_waits_then_succeeds():
    replies = [httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(429),
               httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})]
    waits = []
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=_client(lambda r: replies.pop(0)),
                          rate_limit_retries=2, sleep=waits.append)
    assert llm.complete([{"role": "user", "content": "hi"}]).text == "ok"
    assert waits == [3.0, 8.0]


def test_rate_limit_gives_up_after_retries():
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=_client(lambda r: httpx.Response(429)),
                          rate_limit_retries=1, sleep=lambda s: None)
    with pytest.raises(LLMError, match="429"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_groq_sets_reasoning_effort_per_model():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    groq("k", "qwen/qwen3.8-27b", client=_client(handler)).complete([{"role": "user", "content": "hi"}])
    assert seen["reasoning_effort"] == "none" and seen["model"] == "qwen/qwen3.8-27b"


def test_rate_limit_wait_is_capped():
    replies = [httpx.Response(429, headers={"retry-after": "30"}),
               httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})]
    waits = []
    llm = OpenAICompatLLM("https://x.test/v1", "m", client=_client(lambda r: replies.pop(0)),
                          rate_limit_retries=1, sleep=waits.append, max_wait=3.0)
    llm.complete([{"role": "user", "content": "hi"}])
    assert waits == [3.0]
