import os, re, hashlib, time, json, glob
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from postgrest import PostgrestClient

# ---- Env ----
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE"]
REPO = os.environ.get("RMS_REPO","personal-agent-ste")
BRANCH = os.environ.get("RMS_BRANCH","main")
PREFIX = os.environ.get("RMS_PREFIX","").strip()
RESET = os.environ.get("RMS_RESET","")
SKIP = os.environ.get("RMS_SKIP","backend/static/assets/**,frontend/dist/**,**/node_modules/**")
ROOT = Path(os.environ.get("RMS_ROOT", "."))

EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 80

print(f"[indexer] Indexing with repo={REPO} branch={BRANCH} prefix='{PREFIX}'")
print(f"[indexer] Root={ROOT}")

# ---- Clients ----
oai = OpenAI(api_key=OPENAI_API_KEY)
pg = PostgrestClient(f"{SUPABASE_URL}/rest/v1", headers={
    "apikey": SUPABASE_SERVICE_ROLE,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
    "Content-Type": "application/json",
    "Accept-Profile": "public",
    "Prefer": "return=representation",
})

# ---- Helpers ----
SKIP_GLOBS = [s.strip() for s in SKIP.split(",") if s.strip()]
def should_skip(p: Path) -> bool:
    s = str(p).replace("\\","/")
    for pat in SKIP_GLOBS:
        if glob.fnmatch.fnmatch(s, pat):
            return True
    return False

TEXT_EXTS = {".py",".js",".ts",".json",".md",".txt",".vue",".html",".css",".sh",".ini",".toml",".yml",".yaml",".env",".cfg"}
def is_text_file(p: Path) -> bool:
    return p.suffix.lower() in TEXT_EXTS

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def chunk_text(s: str, max_len: int = 4000) -> List[str]:
    # simple char-based chunking
    blocks = []
    i = 0
    while i < len(s):
        j = min(len(s), i + max_len)
        blocks.append(s[i:j])
        i = j - min(j - i, 500)  # overlap
        if i < 0: i = 0
        if i >= len(s): break
    # ensure progress
    out = []
    last_end = 0
    for blk in blocks:
        if not blk: continue
        end = last_end + len(blk)
        out.append(blk)
        last_end = end
    return out or [s[:max_len]]

def embed_batch(texts: List[str]) -> List[List[float]]:
    clean = [t.replace("\0"," ")[:10000] for t in texts]
    # guard max input size per request
    out = []
    B = 8
    for i in range(0, len(clean), B):
        batch = clean[i:i+B]
        resp = oai.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

# ---- Reset (optional) ----
if RESET:
    print("[indexer] Clearing existing rows…")
    pg.table("repo_chunks").delete().eq("repo", REPO).eq("branch", BRANCH).execute()
    pg.table("repo_files").delete().eq("repo", REPO).eq("branch", BRANCH).execute()

# ---- Walk files ----
paths = []
for p in ROOT.rglob("*"):
    if p.is_dir(): continue
    if PREFIX and not str(p).replace("\\","/").startswith(str((ROOT / PREFIX).as_posix())):
        continue
    if should_skip(p): continue
    if not is_text_file(p): continue
    rel = p.relative_to(ROOT).as_posix()
    paths.append((p, rel))

print(f"[indexer] Indexing {len(paths)} files...")

# ---- Upsert files + chunks ----
files_count = 0
chunks_count = 0
for p, rel in paths:
    try:
        data = p.read_bytes()
    except Exception:
        continue
    text = ""
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        continue

    file_sha = sha1_bytes(data)
    # upsert file row
    pg.table("repo_files").upsert({
        "repo": REPO,
        "branch": BRANCH,
        "path": rel,
        "file_sha": file_sha,
    }, on_conflict="repo,branch,path").execute()

    # chunk & embed
    parts = chunk_text(text, max_len=8000)  # keep under ~8k chars
    vecs = embed_batch(parts)

    rows = []
    for idx, (chunk, emb) in enumerate(zip(parts, vecs), start=1):
        chunk_sha = sha1_bytes((rel + "::" + str(idx) + "::" + chunk).encode("utf-8"))
        rows.append({
            "repo": REPO,
            "branch": BRANCH,
            "path": rel,
            "file_sha": file_sha,
            "chunk_sha": chunk_sha,
            "start_line": 1,        # optional if you don’t track lines
            "end_line": max(1, chunk.count("\n")+1),
            "content": chunk,
            "embedding_1536": emb,  # IMPORTANT: your table must have this column
            "commit_sha": "HEAD",
        })

    # Insert in small batches
    for i in range(0, len(rows), 100):
        pg.table("repo_chunks").upsert(rows[i:i+100],
            on_conflict="repo,branch,path,chunk_sha"
        ).execute()

    files_count += 1
    chunks_count += len(rows)
    print(f"[indexer] OK {rel} ({len(rows)} chunks)")

print(f"[indexer] Done. Files: {files_count}  Chunks: {chunks_count}")
