#!/usr/bin/env python3
"""
CI indexer for RMS.

- Scans the repo (optionally with a path prefix and skip globs)
- Chunks files
- Embeds chunks with OpenAI embeddings
- Upserts into Supabase `repo_memory` (or a table you choose)

To avoid 409 Conflict on insert, this script:
1) Pulls existing (path, chunk_sha) for the repo/branch from Supabase (paged)
2) Skips rows that already exist
3) De-dupes items inside the current run

Env vars (all optional unless marked *required):
  * OPENAI_API_KEY
  * SUPABASE_URL
  * SUPABASE_SERVICE_ROLE        # or SUPABASE_KEY
    EMBED_MODEL                  # default "text-embedding-3-small"
    RMS_REPO                     # default repo name (e.g. "personal-agent-ste")
    RMS_BRANCH                   # default "main"
    RMS_PREFIX                   # e.g. "backend/" — only index files under this
    RMS_SKIP                     # comma-separated globs to ignore
    RMS_ROOT                     # working directory; default "."
    RMS_RESET                    # if set (non-empty), delete existing rows first
    RMS_TABLE                    # default "repo_memory"
    RMS_ON_CONFLICT              # e.g. "repo,branch,path,chunk_sha" (optional)
    RMS_COMMIT_SHA               # optional; stored as commit_sha column if provided
"""

from __future__ import annotations

import os
import io
import sys
import json
import hashlib
from typing import Dict, List, Tuple, Set
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from fnmatch import fnmatch

# -------------------- Config --------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE", "").strip() or os.getenv("SUPABASE_KEY", "").strip()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()

RMS_REPO = os.getenv("RMS_REPO", "personal-agent-ste").strip()
RMS_BRANCH = os.getenv("RMS_BRANCH", "main").strip()
RMS_PREFIX = (os.getenv("RMS_PREFIX") or "").strip()
RMS_SKIP = [g.strip() for g in (os.getenv("RMS_SKIP") or "backend/static/assets/**,frontend/dist/**,**/node_modules/**").split(",") if g.strip()]
RMS_ROOT = (os.getenv("RMS_ROOT") or ".").strip()
RMS_RESET = (os.getenv("RMS_RESET") or "").strip()

RMS_TABLE = os.getenv("RMS_TABLE", "repo_memory").strip()
RMS_ON_CONFLICT = (os.getenv("RMS_ON_CONFLICT") or "").strip()  # e.g., "repo,branch,path,chunk_sha"
RMS_COMMIT_SHA = (os.getenv("RMS_COMMIT_SHA") or "").strip()

# Chunking params (simple & robust)
CHUNK_LINES = 120
CHUNK_OVERLAP = 15
BATCH_EMBED = 128
BATCH_INSERT = 256

# -------------------- HTTP helper --------------------

def _rest(path: str, *, method: str = "GET", body: bytes | None = None, headers: Dict[str, str] | None = None, raise_for_status: bool = True):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/SUPABASE_KEY")

    url = f"{SUPABASE_URL}{path}"
    base_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json",
    }
    if body is not None:
        base_headers["Content-Type"] = "application/json"
    if headers:
        base_headers.update(headers)

    req = Request(url, data=body, method=method, headers=base_headers)
    try:
        with urlopen(req) as resp:
            data = resp.read()
            if not data:
                return None
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                try:
                    return json.loads(data.decode("utf-8"))
                except Exception:
                    return data
            return data
    except HTTPError as he:
        detail = ""
        try:
            detail = he.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        if raise_for_status:
            raise RuntimeError(f"HTTP {he.code} {he.reason} at {url}\n{detail}")
        return {"_error": he.code, "_detail": detail}
    except URLError as ue:
        if raise_for_status:
            raise RuntimeError(f"URL error at {url}: {ue}")
        return {"_error": "urlerror", "_detail": str(ue)}

# -------------------- Filesystem scan --------------------

_TEXT_EXTS = {
    ".py",".js",".ts",".tsx",".jsx",".vue",
    ".json",".toml",".yml",".yaml",".ini",".cfg",
    ".md",".txt",".sql",".sh",".bat",
    ".html",".css",".scss",".rs",".go",".java",".kt",".rb",".php",".c",".h",".cpp",".hpp",
    ".env",".conf",
}

def _should_skip(relpath: str) -> bool:
    if RMS_PREFIX and not relpath.startswith(RMS_PREFIX):
        return True
    for g in RMS_SKIP:
        if fnmatch(relpath, g):
            return True
    _, ext = os.path.splitext(relpath)
    if ext.lower() not in _TEXT_EXTS:
        # still allow README/license etc without extension
        base = os.path.basename(relpath).lower()
        if not (base.startswith("readme") or base in ("license","licence",".gitignore",".dockerignore")):
            return True
    return False

def _walk_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if _should_skip(rel):
                continue
            out.append(rel)
    return out

# -------------------- Chunking --------------------

def _sha1(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def chunk_file(text: str) -> List[Tuple[int,int,str]]:
    """
    Returns list of (start_line, end_line, chunk_text), 1-based inclusive lines.
    """
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    chunks = []
    start = 0
    while start < n:
        end = min(n, start + CHUNK_LINES)
        chunk = "\n".join(lines[start:end])
        chunks.append((start+1, end, chunk))
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks

# -------------------- OpenAI embeddings --------------------

def embed_batch(texts: List[str]) -> List[List[float]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"openai package not available: {e}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    safe_inputs = [(t if isinstance(t, str) and t.strip() else " ") for t in texts]
    resp = client.embeddings.create(model=EMBED_MODEL, input=safe_inputs)
    return [item.embedding for item in resp.data]

# -------------------- Supabase helpers --------------------

def fetch_existing_keys(repo: str, branch: str) -> Set[Tuple[str, str]]:
    """
    Fetch existing (path, chunk_sha) pairs for the repo/branch.
    Uses proper PostgREST pagination headers to avoid 409.
    """
    keys: Set[Tuple[str, str]] = set()
    page = 0
    page_size = 5000

    while True:
        rng_start = page * page_size
        rng_end = rng_start + page_size - 1
        res = _rest(
            f"/rest/v1/{RMS_TABLE}?repo=eq.{repo}&branch=eq.{branch}&select=path,chunk_sha",
            headers={
                "Range-Unit": "items",
                "Range": f"{rng_start}-{rng_end}",
            },
            method="GET",
            body=None,
            raise_for_status=False,
        )
        if isinstance(res, dict) and "_error" in res:
            # If table is empty or permissions are restrictive, break cleanly
            break
        rows = res or []
        if not rows:
            break
        for r in rows:
            p = (r.get("path") or "").strip()
            s = (r.get("chunk_sha") or "").strip()
            if p and s:
                keys.add((p, s))
        if len(rows) < page_size:
            break
        page += 1

    return keys

def delete_repo_branch(repo: str, branch: str):
    _rest(
        f"/rest/v1/{RMS_TABLE}?repo=eq.{repo}&branch=eq.{branch}",
        method="DELETE",
        body=None,
    )

def insert_rows(rows: List[Dict]):
    """
    Insert rows in batches; skip upsert unless RMS_ON_CONFLICT is provided AND
    the DB has a matching unique/primary key. Prefiltering avoids 409s.
    """
    if not rows:
        return
    q = f"/rest/v1/{RMS_TABLE}"
    if RMS_ON_CONFLICT:
        q += f"?on_conflict={RMS_ON_CONFLICT}"

    i = 0
    while i < len(rows):
        batch = rows[i:i+BATCH_INSERT]
        body = json.dumps(batch).encode("utf-8")
        _rest(q, method="POST", body=body)
        i += len(batch)

# -------------------- Main flow --------------------

def main():
    print(f"[indexer] Indexing with repo={RMS_REPO} branch={RMS_BRANCH} prefix='{RMS_PREFIX}'", flush=True)
    root = RMS_ROOT or "."
    print(f"[indexer] Root={root}", flush=True)

    all_files = _walk_files(root)
    print(f"[indexer] Indexing {len(all_files)} files…", flush=True)

    if RMS_RESET:
        print("[indexer] Clearing existing rows…", flush=True)
        delete_repo_branch(RMS_REPO, RMS_BRANCH)

    # Pull existing keys to avoid 409s
    existing = fetch_existing_keys(RMS_REPO, RMS_BRANCH)
    print(f"[indexer] Found {len(existing)} existing (path, chunk_sha) rows in DB", flush=True)

    # Prepare chunks to embed
    to_embed_texts: List[str] = []
    to_embed_meta: List[Tuple[str, int, int, str]] = []  # (path, start, end, chunk_sha)
    seen_in_run: Set[Tuple[str, str]] = set()

    for rel in all_files:
        full = os.path.join(root, rel)
        try:
            with io.open(full, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"[indexer] SKIP {rel} (read error: {e})", flush=True)
            continue

        chunks = chunk_file(text)
        if not chunks:
            continue

        print(f"[indexer] OK {rel} ({len(chunks)} chunks)", flush=True)

        for (start, end, chunk_text) in chunks:
            csha = _sha1(chunk_text)
            key = (rel, csha)
            if key in existing:
                continue  # already present in DB — skip
            if key in seen_in_run:
                continue  # avoid duplicates in the same CI run
            seen_in_run.add(key)
            to_embed_texts.append(chunk_text)
            to_embed_meta.append((rel, start, end, csha))

    if not to_embed_texts:
        print("[indexer] Nothing new to index (all chunks already exist).", flush=True)
        return

    # Embed in batches
    vectors: List[List[float]] = []
    idx = 0
    while idx < len(to_embed_texts):
        batch = to_embed_texts[idx:idx+BATCH_EMBED]
        vecs = embed_batch(batch)
        vectors.extend(vecs)
        idx += len(batch)

    if len(vectors) != len(to_embed_meta):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} vs {len(to_embed_meta)}")

    # Build rows
    rows: List[Dict] = []
    for (meta, vec) in zip(to_embed_meta, vectors):
        path, start, end, csha = meta
        rows.append({
            "repo": RMS_REPO,
            "branch": RMS_BRANCH,
            "path": path,
            "start_line": int(start),
            "end_line": int(end),
            "content": None,            # store only embeddings & metadata; content optional
            "embedding": vec,
            "chunk_sha": csha,
            "commit_sha": RMS_COMMIT_SHA or None,
        })

    # Insert rows (no 409; we prefiltered)
    insert_rows(rows)
    print(f"[indexer] Inserted {len(rows)} new rows.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Traceback (most recent call last):\n{e}", file=sys.stderr)
        raise
