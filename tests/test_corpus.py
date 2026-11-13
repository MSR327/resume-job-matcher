import pandas as pd

from jobmatch.corpus import build_corpus, clean_text, is_tech_title


def test_tech_titles():
    for t in ["Senior Software Engineer", "Data Scientist II", "ML Engineer", "DevOps Engineer",
              "Frontend Developer (React)", "Site Reliability Engineer", "QA Automation Engineer"]:
        assert is_tech_title(t), t
    for t in ["Sales Engineer", "Registered Nurse", "Mechanical Engineer", "Technical Recruiter",
              "Civil Engineer", "Account Executive"]:
        assert not is_tech_title(t), t


def test_clean_text_collapses_whitespace_and_bullets():
    assert clean_text("Hello  world\n\n\n• Python\r\n") == "Hello world\n\n- Python"


def _row(i, title, desc, company="Acme"):
    return {"job_id": i, "title": title, "company_name": company, "location": "Remote",
            "description": desc, "job_posting_url": f"https://x/{i}", "listed_time": 1700000000000}


def test_build_corpus_filters_and_dedupes():
    long = "Build APIs in Python and Go. " * 30
    df = pd.DataFrame([
        _row(1, "Software Engineer", long),
        _row(2, "Software Engineer", long),  # duplicate (same company+title+desc)
        _row(3, "Software Engineer", "too short"),  # short description
        _row(4, "Registered Nurse", long),  # not tech
        _row(5, "Data Engineer", long, company="Beta"),
    ])
    out = build_corpus(df)
    assert out["job_id"].tolist() == ["1", "5"]
    assert set(out.columns) == {"job_id", "title", "company", "location", "description", "url", "listed_at"}
