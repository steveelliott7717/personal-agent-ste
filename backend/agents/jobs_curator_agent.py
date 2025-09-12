# backend/agents/jobs_curator_agent.py
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from backend.registry.capability_registry import flatten_result

if TYPE_CHECKING:
    from backend.registry.capability_registry import CapabilityRegistry

logger = logging.getLogger(__name__)


def _parse_job_board_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a URL to extract the job board platform and external ID.
    Supports Greenhouse, Lever, and Ashby.
    """
    if not isinstance(url, str):
        return None, None

    # Greenhouse: boards.greenhouse.io/company/jobs/12345
    m = re.search(r"boards\.greenhouse\.io/([^/]+)/jobs/(\d+)", url)
    if m:
        return "greenhouse", m.group(2)

    # Lever: jobs.lever.co/company/abc-def-123
    m = re.search(r"jobs\.lever\.co/([^/]+)/([0-9a-f-]+)", url)
    if m:
        return "lever", m.group(2)

    # Ashby: jobs.ashbyhq.com/company/abc-def-123
    m = re.search(r"jobs\.ashbyhq\.com/([^/]+)/([0-9a-f-]+)", url)
    if m:
        return "ashby", m.group(2)

    return None, None


def run_jobs_curator(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads from jobs_normalized, enriches, and upserts into the final `jobs` table.
    This is a deterministic function using only the capability registry.
    """
    from backend.registry.capability_registry import CapabilityRegistry

    registry = CapabilityRegistry()
    run_id = args.get("run_id")
    limit = int(args.get("limit", 200))
    correlation_id = meta.get("correlation_id", f"curator-{run_id}")

    # 1. Read recent normalized jobs
    read_args = {
        "table": "jobs_normalized",
        "select": "source,url,title,company,location,work_mode,created_at",
        "order": {"field": "created_at", "desc": True},
        "limit": limit,
    }
    read_res = registry.dispatch("db.read", read_args, meta)
    if not read_res.get("ok"):
        return {"ok": False, "error": read_res.get("error")}

    normalized_jobs = flatten_result(read_res).get("rows", [])
    if not normalized_jobs:
        return {"ok": True, "finish": {"upserted": 0, "seen": 0}}

    # 2. Build upsert rows for `public.jobs`
    jobs_to_upsert: List[Dict[str, Any]] = []
    urls_seen = set()

    for job in normalized_jobs:
        url = job.get("url")
        if not url or url in urls_seen:
            continue
        urls_seen.add(url)

        board, external_id = _parse_job_board_from_url(url)

        upsert_row = {
            "url": url,
            "board": board,
            "external_id": external_id,
            "title": job.get("title"),
            "org": job.get("company"),
            "location": job.get("location"),
            "last_seen_at": "now()",
            "source": job.get("source"),
            "run_id": run_id,
            "status": "open",
            "remote": job.get("work_mode", "").lower() == "remote",
        }
        jobs_to_upsert.append(upsert_row)

    if not jobs_to_upsert:
        return {"ok": True, "finish": {"upserted": 0, "seen": len(normalized_jobs)}}

    # 3. Upsert into `public.jobs`
    write_args = {
        "table": "jobs",
        "rows": jobs_to_upsert,
        "mode": "upsert",
        "on_conflict": "url",
        "returning": "minimal",
    }

    idem_key = f"{correlation_id}-write-{len(jobs_to_upsert)}"
    write_meta = {**meta, "idempotency_key": idem_key}

    write_res = registry.dispatch("db.write", write_args, write_meta)

    if not write_res.get("ok"):
        return {"ok": False, "error": write_res.get("error")}

    upserted_count = len(jobs_to_upsert)

    return {
        "ok": True,
        "finish": {
            "upserted": upserted_count,
            "seen": len(normalized_jobs),
        },
    }


def handle_jobs_curation(query: str | Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent entrypoint for the jobs curator.
    It's a deterministic agent that doesn't use an LLM.
    """
    logger.info(f"JobsCuratorAgent triggered with query: {query}")

    args = {}
    if isinstance(query, dict):
        args = query

    run_id = args.get("run_id")
    meta = {"correlation_id": f"curator-handler-{run_id}"}

    try:
        result = run_jobs_curator(args, meta)
        return result
    except Exception as e:
        logger.exception(f"JobsCuratorAgent failed: {e}")
        return {"ok": False, "error": {"code": "AgentFailure", "message": str(e)}}
