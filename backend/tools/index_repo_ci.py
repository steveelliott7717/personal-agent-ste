# backend/tools/index_repo_ci.py
import os
import re
import hashlib
import json
import glob
from pathlib import Path
from typing import List
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote

from openai import OpenAI

# -------- Env --------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE = os.environ.get("SUPABASE_SERVICE_ROLE", "")
REPO = os.environ.get("RMS_REPO", "personal-agent-ste")
BRANCH = os.environ.get("RMS_BRANCH", "main")
PREFIX = (os.environ.get("RMS_PREFIX", "") or "").strip()  # e.g. "backend/" or ""
RESET = os.environ.get("RMS_RESET", "")
SKIP = os.environ.get("RMS_SKIP", "backend/static/assets/**,frontend/dist/**,**/node_modules/**")
ROOT = Path(os.environ.get("RMS_ROOT", "."))

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims

print(f"[indexer] Indexing with repo={REPO} branch={BRANCH} prefix='{PREFIX}'")
print(f"[indexer] Root={ROOT}")

# -------- Clients --------
if not OPENAI_API_KEY:
    print("[indexer] WARNING: OPENAI_API_KEY is empty")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
    print("[indexer] WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE is empty")

oai = OpenAI(api_key=OPENAI_API_KEY)

# -------- REST helper --------
def _rest(method: str, path: str, body: dict | list | None = None, params: dict | None = None):
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL not set")
    url = f"{SUPABASE_URL}{path}"
    if params:
        qs = urlencode(params, doseq=True)
        url = f"{url}?{qs}"
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
    }

    req = Request(url, data=data, headers=headers, method=method.upper())
    with urlopen(req) as resp:
        payload = resp.read()
    return json.loads(payload) if payload else None

# -------- Patterns / helpers --------
SKIP_GLOBS = [s.strip() for s in SKIP.split(",") if s.strip()]

def should_skip(p: Path) -> bool:
    s = str(p).replace("\\", "/")
    for pat in SKIP_GLOBS:
        if glob.fnmatch.fnmatch(s, pat):
            return True
    return False

TEXT_EXTS = {
    ".py", ".js", ".ts", ".json", ".md", ".txt", ".vue", ".html", ".css",
    ".sh", ".ini", ".toml", ".yml", ".yaml", ".env", ".cfg"
}

def is_text_file(p: Path) -> bool:
    return p.suffix.lower() in TEXT_EXTS

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def chunk_text(s: str, max_len: int = 8000, overlap: int = 500) -> List[str]:
    # simple char-based overlapping chunker
    if not s:
        return [""]
    blocks = []
    start = 0
    n = len(s)
    while start < n:
        end = min(n, start + max_len)
        blocks.append(s[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return blocks

def embed_batch(texts: List[str]) -> List[List[float]]:
    cleaned = [t.replace("\0", " ")[:10000] for t in texts]  # guard very long inputs
    out: List[List[float]] = []
    B = 8
    for i in range(0, len(cleaned), B):
        batch = cleaned[i:i+B]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

# -------- Optional reset --------
if RESET:
    print("[indexer] Clearing existing rows…")
    # DELETE from repo_chunks where repo=… and branch=…
    _rest(
        "DELETE",
        "/rest/v1/repo_chunks",
        params={"repo": f"eq.{REPO}", "branch": f"eq.{BRANCH}"},
    )
    _rest(
        "DELETE",
        "/rest/v1/repo_files",
        params={"repo": f"eq.{REPO}", "branch": f"eq.{BRANCH}"},
    )

# -------- Gather files --------
paths: List[tuple[Path, str]] = []
prefix_dir = (ROOT / PREFIX).resolve() if PREFIX else None

for p in ROOT.rglob("*"):
    if p.is_dir():
        continue
    # Prefix filter
    if prefix_dir and not str(p.resolve()).startswith(str(prefix_dir)):
        continue
    if should_skip(p):
        continue
    if not is_text_file(p):
        continue
    rel = p.relative_to(ROOT).as_posix()
    paths.append((p, rel))

print(f"[indexer] Indexing {len(paths)} files…")

# -------- Upsert files + chunks --------
files_count = 0
chunks_count = 0

for p, rel in paths:
    try:
        data = p.read_bytes()
    except Exception as e:
        print(f"[indexer] SKIP {rel}: read failed: {e}")
        continue

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[indexer] SKIP {rel}: decode failed: {e}")
        continue

    file_sha = sha1_bytes(data)

    # Upsert repo_files (on conflict repo,branch,path)
    _rest(
        "POST",
        "/rest/v1/repo_files",
        body=[{
            "repo": REPO,
            "branch": BRANCH,
            "path": rel,
            "file_sha": file_sha,
        }],
        params={"on_conflict": "repo,branch,path"},
    )

    # Chunk & embed
    parts = chunk_text(text, max_len=8000, overlap=500)
    vecs = embed_batch(parts)

    rows = []
    for idx, (chunk, emb) in enumerate(zip(parts, vecs), start=1):
        chunk_sha = sha1_bytes((rel + "::" + str(idx) + "::" + chunk).encode("utf-8"))
        # We don't have exact line mapping here; store approximate end_line
        rows.append({
            "repo": REPO,
            "branch": BRANCH,
            "path": rel,
            "file_sha": file_sha,
            "chunk_sha": chunk_sha,
            "start_line": 1,
            "end_line": max(1, chunk.count("\n") + 1),
            "content": chunk,
            "embedding_1536": emb,  # ensure this column exists in your table
            "commit_sha": "HEAD",
        })

    # Upsert in batches into repo_chunks (on conflict repo,branch,path,chunk_sha)
    for i in range(0, len(rows), 100):
        _rest(
            "POST",
            "/rest/v1/repo_chunks",
            body=rows[i:i+100],
            params={"on_conflict": "repo,branch,path,chunk_sha"},
        )

    files_count += 1
    chunks_count += len(rows)
    print(f"[indexer] OK {rel} ({len(rows)} chunks)")

print(f"[indexer] Done. Files: {files_count}  Chunks: {chunks_count}")
