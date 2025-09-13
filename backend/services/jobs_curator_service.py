# backend/services/jobs_curator_service.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from backend.services.supabase_service import supabase
from backend.utils.jobs_parsing import (
    parse_board_external_id,
    extract_listed_at,
    brief_excerpt,
)


def _rows_to_map(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for r in rows or []:
        k = r.get(key)
        if k:
            m[k] = r
    return m


def curate_jobs(
    run_id: Optional[str] = None, normalized_limit: int = 250
) -> Dict[str, Any]:
    """
    Read recent jobs_normalized, enrich from jobs_scores and fetched_pages, and upsert curated rows into public.jobs.
    - Upsert by URL first
    - For rows where (board, external_id) can be parsed, also upsert by that pair (second pass)
    """
    # 1) Read recent normalized jobs
    jn = (
        supabase.table("jobs_normalized")
        .select("source,url,title,company,location,work_mode,raw_json,created_at")
        .order("created_at", desc=True)
        .limit(normalized_limit)
        .execute()
        .data
        or []
    )

    if not jn:
        return {"finish": {"seen": 0, "upserted": 0}}

    # 2) Enrich: latest scores by (source,url)
    urls = [r["url"] for r in jn if r.get("url")]
    # dedupe for IN query
    url_set = sorted(list({u for u in urls if u}))
    scores = (
        supabase.table("jobs_scores")
        .select("source,url,score,created_at")
        .in_("url", url_set)
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
        .data
        or []
    )
    # Keep only the latest score per (source,url)
    latest_score: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in scores:
        key = (row.get("source") or "", row.get("url") or "")
        if key and (
            key not in latest_score
            or latest_score[key]["created_at"] < row["created_at"]
        ):
            latest_score[key] = row

    # 3) Enrich: fetched_pages sample_text for description excerpt / listed_at fallback
    fp = (
        supabase.table("fetched_pages")
        .select("url,sample_text")
        .in_("url", url_set)
        .order("fetched_at", desc=True)
        .limit(2000)
        .execute()
        .data
        or []
    )
    fp_by_url = _rows_to_map(fp, "url")

    now_utc = datetime.now(timezone.utc)
    upsert_rows: List[Dict[str, Any]] = []

    for row in jn:
        url = row.get("url")
        if not url:
            continue

        source = row.get("source")
        title = row.get("title")
        org = row.get("company") or row.get("org")
        location = row.get("location")
        work_mode = (row.get("work_mode") or "").lower().strip()

        board, external_id = parse_board_external_id(url)

        sample = fp_by_url.get(url, {}).get("sample_text")
        description_excerpt = brief_excerpt(sample, 800)

        # Try listed_at from text if reasonable:
        listed_at = extract_listed_at(sample)
        # If we can't parse, leave null; don't fabricate.

        # Map automation_fit from latest jobs_scores if available
        score_row = latest_score.get((source or "", url), None)
        automation_fit = score_row.get("score") if score_row else None

        curated = {
            "url": url,
            "source": source,
            "title": title,
            "org": org,
            "location": location,
            "description_excerpt": description_excerpt,
            "listed_at": listed_at.isoformat() if listed_at else None,
            "last_seen_at": now_utc.isoformat(),
            "status": "open",  # IMPORTANT: treat curated jobs as open by default
            "run_id": run_id,
            "automation_fit": automation_fit,
        }

        # Prefer explicit remote flag when work_mode says remote
        if work_mode == "remote":
            curated["remote"] = True

        # Best-effort board/external_id
        if board:
            curated["board"] = board
        if external_id:
            curated["external_id"] = external_id

        upsert_rows.append(curated)

    if not upsert_rows:
        return {"finish": {"seen": len(jn), "upserted": 0}}

    # 4) Upsert by URL
    # NOTE: Supabase python client uses PostgREST; emulate "on_conflict" via RPC or
    # pass "upsert=True" with "ignore_duplicates=False" and specify a conflict target if available.
    # If your helper uses a generic wrapper, adapt accordingly.
    upserted = 0
    batch_size = 200
    for i in range(0, len(upsert_rows), batch_size):
        batch = upsert_rows[i : i + batch_size]
        (
            supabase.table("jobs")
            .upsert(batch, on_conflict="url", returning="minimal")
            .execute()
        )
        # PostgREST doesn't return affected row count with returning=minimal; treat as success
        upserted += len(batch)

    # 5) Second pass: upsert again where (board,external_id) present — helps dedupe across boards
    be_rows = [r for r in upsert_rows if r.get("board") and r.get("external_id")]
    if be_rows:
        for i in range(0, len(be_rows), batch_size):
            batch = be_rows[i : i + batch_size]
            (
                supabase.table("jobs")
                .upsert(batch, on_conflict="board,external_id", returning="minimal")
                .execute()
            )

    return {"finish": {"seen": len(jn), "upserted": upserted}}
