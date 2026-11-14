from jobmatch.judge import judge
from jobmatch.llm import FakeLLM, LLMError

JOB = {"title": "ML Engineer", "description": "PyTorch, Docker."}


def test_judge_parses_grade():
    assert judge(FakeLLM(['{"reason": "fits", "grade": 3}']), "r", JOB) == 3
    assert judge(FakeLLM(['{"reason": "x", "grade": "2"}']), "r", JOB) == 2


def test_judge_retries_out_of_range_then_none():
    assert judge(FakeLLM(['{"grade": 7}', '{"grade": 1}']), "r", JOB) == 1
    assert judge(FakeLLM([LLMError("x"), "nope"]), "r", JOB) is None
