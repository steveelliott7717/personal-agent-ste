# backend/utils/jobs_parsing.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

# Common ATS URL patterns:
# - Greenhouse: https://boards.greenhouse.io/<org>/jobs/<id>
GH_RE = re.compile(
    r"https?://boards\.greenhouse\.io/([^/]+)/jobs/([a-zA-Z0-9\-]+)", re.I
)

# - Lever: https://jobs.lever.co/<org>/<id or slug> or https://jobs.lever.co/<org>/<id>/apply
LEVER_RE = re.compile(r"https?://jobs\.lever\.co/([^/]+)/([a-zA-Z0-9\-]+)", re.I)

# - Ashby: https://jobs.ashbyhq.com/<org>/<slug> (optionally ending with /apply or query)
ASHBY_RE = re.compile(r"https?://jobs\.ashbyhq\.com/([^/]+)/([a-zA-Z0-9\-%]+)", re.I)

# A very light-weight date finder (RFC-ish or simple Month Day, Year). Best effort only.
DATE_RE = re.compile(
    r"(?P<iso>\b\d{4}-\d{2}-\d{2}\b)"
    r"|(?P<mdy>\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b)",
    re.I,
)


def parse_board_external_id(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to parse (board, external_id) from a known ATS job URL.
    Returns (board, external_id) or (None, None) if not recognized.
    """
    if not url:
        return None, None

    m = GH_RE.search(url)
    if m:
        org, jobid = m.group(1), m.group(2)
        return ("greenhouse", f"{org}:{jobid}")

    m = LEVER_RE.search(url)
    if m:
        org, jobid = m.group(1), m.group(2)
        return ("lever", f"{org}:{jobid}")

    m = ASHBY_RE.search(url)
    if m:
        org, slug = m.group(1), m.group(2)
        return ("ashby", f"{org}:{slug}")

    return (None, None)


def extract_listed_at(sample_text: Optional[str]) -> Optional[datetime]:
    """
    Try to find a plausible 'posted/listed' date in text. Best effort.
    Returns a timezone-naive datetime (UTC assumed) or None.
    """
    if not sample_text:
        return None

    m = DATE_RE.search(sample_text)
    if not m:
        return None

    iso = m.group("iso")
    mdy = m.group("mdy")

    try:
        if iso:
            return datetime.strptime(iso, "%Y-%m-%d")
        if mdy:
            return datetime.strptime(mdy, "%b %d, %Y")
    except Exception:
        return None

    return None


def brief_excerpt(text: Optional[str], max_len: int = 800) -> Optional[str]:
    """
    Return a short, single-line excerpt up to max_len.
    """
    if not text:
        return None
    cleaned = " ".join(text.split())  # collapse whitespace
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"
