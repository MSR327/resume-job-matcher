"""Filter the LinkedIn postings dump to clean, deduplicated tech jobs."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

_INCLUDE = re.compile(
    r"\b(software|developer|programmer|full[- ]?stack|front[- ]?end|back[- ]?end|web|mobile|ios|android|"
    r"data (scientist|engineer|analyst|architect)|machine learning|ml|ai|deep learning|nlp|"
    r"devops|site reliability|sre|cloud|platform engineer|infrastructure engineer|"
    r"security engineer|cyber ?security|application security|qa|quality assurance|test automation|sdet|"
    r"database|dba|analytics engineer|solutions architect|software architect|computer vision)\b",
    re.IGNORECASE,
)
_EXCLUDE = re.compile(
    r"\b(sales|recruit\w*|account (executive|manager)|nurse|mechanical|civil|electrical|"
    r"chemical|manufacturing|marketing|teacher|driver)\b",
    re.IGNORECASE,
)


def is_tech_title(title: str) -> bool:
    title = title or ""
    return bool(_INCLUDE.search(title)) and not _EXCLUDE.search(title)


def clean_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    t = re.sub(r"^[ \t]*[•●▪◦*·][ \t]*", "- ", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def build_corpus(df: pd.DataFrame, min_desc_chars: int = 400) -> pd.DataFrame:
    """Raw postings (arshkon LinkedIn schema) -> tech jobs with a clean, stable schema."""
    out = pd.DataFrame({
        "job_id": df["job_id"].astype(str),
        "title": df["title"].fillna("").map(str.strip),
        "company": df["company_name"].fillna("").map(str.strip),
        "location": df["location"].fillna(""),
        "description": df["description"].fillna("").map(clean_text),
        "url": df["job_posting_url"].fillna(""),
        "listed_at": df["listed_time"].fillna(0).astype("int64"),
    })
    out = out[out["title"].map(is_tech_title) & (out["description"].str.len() >= min_desc_chars)]
    # Same company + title + description = the same job reposted.
    key = (
        out["company"].str.lower() + "|" + out["title"].str.lower() + "|"
        + out["description"].map(lambda d: hashlib.md5(d.lower().encode()).hexdigest())
    )
    out = out.loc[~key.duplicated()].drop_duplicates("job_id")
    return out.reset_index(drop=True)
