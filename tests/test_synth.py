import json
import random

from jobmatch.llm import FakeLLM, LLMError
from jobmatch.synth import make_resume, split_dev_test

JOB = {"job_id": "j1", "title": "Data Engineer",
       "description": "You will build batch and streaming pipelines with Spark and Airflow on AWS. " * 3}
ORIGINAL = ("Data engineer with four years designing ingestion jobs. Moved a nightly warehouse load to "
            "incremental processing, cutting runtime by half. Comfortable in Python, SQL and Scala; "
            "scheduled workflows with an open-source orchestrator; some exposure to cloud object storage. "
            "Earlier worked in retail analytics building dashboards for store managers. Mentored two "
            "interns and wrote internal docs on data quality checks. Looking to deepen real-time skills.")
COPIED = "You will build batch and streaming pipelines with Spark and Airflow on AWS. " * 4


def _r(text):
    return json.dumps({"resume": text})


def test_accepts_paraphrased_resume():
    out = make_resume(FakeLLM([_r(ORIGINAL)]), JOB, random.Random(0))
    assert out and out["job_id"] == "j1" and out["overlap4"] <= 0.10


def test_rejects_copy_then_retries():
    fake = FakeLLM([_r(COPIED), _r(ORIGINAL)])
    out = make_resume(fake, JOB, random.Random(0))
    assert out and len(fake.calls) == 2


def test_gives_up_after_tries():
    assert make_resume(FakeLLM([LLMError("x"), _r("too short")]), JOB, random.Random(0)) is None


def test_split_is_deterministic_and_disjoint():
    items = [{"job_id": str(i)} for i in range(10)]
    d1, t1 = split_dev_test(items, 3, seed=1)
    d2, _ = split_dev_test(list(reversed(items)), 3, seed=1)
    assert d1 == d2 and len(t1) == 7 and not {x["job_id"] for x in d1} & {x["job_id"] for x in t1}
