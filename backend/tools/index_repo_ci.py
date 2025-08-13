# backend/tools/index_repo_ci.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# --------- Config / Env ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
# dims: 3072 for 3-large, otherwise 1536 for 3-small
EMBED_DIMS = 3072 if "3-large" in EMBED_MODEL else 1536

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_APIKEY")
    or os.getenv("OPENAI_KEY")
)
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("Missing SUPABASE_URL and/or SUPABASE_SERVICE_ROLE (or SUPABASE_KEY).", file=sys.stderr)
    sys.exit(1)

# --------- HTTP helpers ----------
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def _rest(url: str, method: str = "GET", body: Optional[dict] = None, supabase_key: Optional[str] = None) -> dict:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if supabase_key:
        req.add_header("apikey", supabase_key)
        req.add_header("Authorization", f"Bearer {supabase_key}")
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {e.reason} at {url}\n{detail}") from None
    except URLError as e:
        raise RuntimeError(f"Network error contacting {url}: {e}") from None


# --------- OpenAI client ----------
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)


def rough_token_estimate(s: str) -> int:
    # ~4 chars per token roughness
    return max(1, math.ceil(len(s) / 4))


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


# Trim per-piece and pack into sub-batches under model limits.
def pack_batches(
    texts: List[str],
    *,
    piece_max_tokens: int,
    request_max_tokens: int,
    request_max_chars: int,
    batch_size: int
) -> List[List[str]]:
    safe: List[str] = []
    per_piece_char_cap = max(100, piece_max_tokens * 4)  # ~4 chars/token
    for s in texts:
        s = "" if s is None else (json.dumps(s) if isinstance(s, (dict, list)) else str(s))
        if len(s) > per_piece_char_cap:
            s = s[:per_piece_char_cap]
        safe.append(s)

    batches: List[List[str]] = []
    i = 0
    N = len(safe)
    while i < N:
        cur: List[str] = []
        cur_tokens = 0
        cur_chars = 0
        while i < N and len(cur) < batch_size:
            s = safe[i]
            t = rough_token_estimate(s)
            if (cur_tokens + t) <= request_max_tokens and (cur_chars + len(s)) <= request_max_chars:
                cur.append(s)
                cur_tokens += t
                cur_chars += len(s)
                i += 1
            else:
                if not cur:
                    # single very large piece — force include (already truncated)
                    cur.append(s)
                    i += 1
                break
        batches.append(cur)
    return batches


def embed_batch_safe(inputs: List[str]) -> List[List[float]]:
    """
    Attempt a single embeddings.create call; on 400 context errors, split and retry recursively.
    """
    if not inputs:
        return []

    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=inputs)
        data = resp.data or []
        return [d.embedding for d in data]
    except Exception as e:
        msg = str(e)
        # If we exceeded context, bisect and embed halves
        if "maximum context length" in msg.lower() and len(inputs) > 1:
            mid = len(inputs) // 2
            left = embed_batch_safe(inputs[:mid])
            right = embed_batch_safe(inputs[mid:])
            return left + right
        raise


# --------- Supabase upsert helpers ----------
def supabase_upsert_rows(rows: list[dict], verbose: bool = False):
    """
    Upserts into:
      - repo_chunks (on_conflict=chunk_sha)
      - repo_memory  (on_conflict=ux_repo_memory_chunk)
    Falls back to plain insert if on_conflict not supported.
    Treats 409 duplicates as success (idempotent).
    """
    if not rows:
        return

    url_chunks = f"{SUPABASE_URL}/rest/v1/repo_chunks?on_conflict=chunk_sha"
    url_memory = f"{SUPABASE_URL}/rest/v1/repo_memory?on_conflict=ux_repo_memory_chunk"

    def _send(url, payload):
        try:
            _rest(url, method="POST", body=payload, supabase_key=SUPABASE_KEY)
        except RuntimeError as e:
            emsg = str(e)
            # 409 duplicate -> success (already present)
            if "http 409" in emsg.lower() and "duplicate key value" in emsg.lower():
                if verbose:
                    print("[db] duplicate rows — continuing")
                return
            # on_conflict not supported -> fall back to plain insert
            if ("on_conflict" in url) and (
                "does not exist" in emsg.lower()
                or "42703" in emsg
                or "pgrst204" in emsg.lower()
            ):
                fallback = url.split("?")[0]
                if verbose:
                    print(f"[db] falling back to plain insert: {fallback}")
                try:
                    _rest(fallback, method="POST", body=payload, supabase_key=SUPABASE_KEY)
                except RuntimeError as e2:
                    # If duplicates appear here, also swallow
                    if "http 409" in str(e2).lower() and "duplicate key value" in str(e2).lower():
                        if verbose:
                            print("[db] duplicate rows (fallback) — continuing")
                        return
                    raise
            else:
                raise

    # repo_chunks rows (minimal schema)
    chunk_rows = [{
        "repo": r["repo"],
        "branch": r["branch"],
        "path": r["path"],
        "start_line": r["start_line"],
        "end_line": r["end_line"],
        "content": r["content"],
        "file_sha": r.get("file_sha"),
        "chunk_sha": r.get("chunk_sha"),
        "commit_sha": r.get("commit_sha"),
        "head_ref": r.get("commit_sha") or "HEAD",
        "chunk_index": r.get("chunk_index"),
    } for r in rows]

    # repo_memory rows (your schema includes repo_name + dims)
    mem_rows_raw = [{
        "repo_name": r["repo"],
        "repo": r["repo"],
        "branch": r["branch"],
        "head_ref": r.get("commit_sha") or "HEAD",
        "path": r["path"],
        "file_sha": r.get("file_sha"),
        "chunk_sha": r.get("chunk_sha"),
        "start_line": r["start_line"],
        "end_line": r["end_line"],
        "content": r["content"],
        "embedding": r["embedding"],
        "dims": r["dims"],
        "commit_sha": r.get("commit_sha"),
    } for r in rows]

    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    mem_rows = [_clean(m) for m in mem_rows_raw]

    _send(url_chunks, chunk_rows)
    _send(url_memory, mem_rows)


# --------- FS + chunking ----------
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", ".next", ".turbo"}
DEFAULT_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".sql", ".json", ".md",
    ".yml", ".yaml", ".toml", ".ini", ".env", ".txt"
}


def file_iter(root: Path, include_ext: set[str], paths: Optional[List[str]]) -> Iterable[Path]:
    base = root.resolve()
    allow_prefixes = []
    if paths:
        for p in paths:
            allow_prefixes.append((base / p).resolve())

    for p in base.rglob("*"):
        if p.is_dir():
            if p.name in SKIP_DIRS:
                continue
        if p.is_file():
            if include_ext and p.suffix.lower() not in include_ext:
                continue
            if allow_prefixes and not any(str(p).startswith(str(ap)) for ap in allow_prefixes):
                continue
            yield p


def read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []


def chunk_lines(lines: List[str], max_lines: int, overlap: int) -> Iterable[Tuple[int, int, str]]:
    n = len(lines)
    if n == 0:
        return
    i = 0
    while i < n:
        j = min(n, i + max_lines)
        start, end = i + 1, j  # 1-based inclusive
        text = "\n".join(lines[i:j])
        yield start, end, text
        if j >= n:
            break
        i = j - overlap if overlap > 0 else j


# --------- Incremental map ----------
def load_existing_file_shas(repo: str, branch: str) -> Dict[Tuple[str, str, str], str]:
    """
    Query repo_chunks for existing (repo, branch, path) -> file_sha mapping.
    """
    url = f"{SUPABASE_URL}/rest/v1/repo_chunks?select=repo,branch,path,file_sha"
    rows = _rest(url, method="GET", supabase_key=SUPABASE_KEY)
    out: Dict[Tuple[str, str, str], str] = {}
    for r in rows:
        key = (r.get("repo"), r.get("branch"), r.get("path"))
        fs = r.get("file_sha")
        if key[0] and key[1] and key[2] and fs:
            out[key] = fs
    return out


# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Index repo into Supabase with embeddings (with incremental mode).")
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--paths", default="", help='Quoted space-separated list, e.g. "backend frontend/src file.js"')
    ap.add_argument("--include-ext", default=",".join(sorted(DEFAULT_EXTS)))
    ap.add_argument("--max-lines", type=int, default=80)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--piece-max-tokens", type=int, default=5200)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--incremental", action="store_true", help="Skip files whose file_sha matches existing repo_chunks")
    # internal-ish caps for request packing (safe for 8k ctx)
    ap.add_argument("--request-max-chars", type=int, default=10000)
    ap.add_argument("--request-max-tokens", type=int, default=6500)

    args = ap.parse_args()

    repo = args.repo_name
    branch = args.branch
    root = Path(args.root).resolve()
    include_ext = {e.strip().lower() for e in args.include_ext.split(",") if e.strip()}
    paths = [p for p in args.paths.split() if p.strip()] if args.paths else None

    # Build file list
    files = list(file_iter(root, include_ext, paths))
    total_files = len(files)

    # Optional incremental map
    existing_map: Dict[Tuple[str, str, str], str] = {}
    if args.incremental:
        try:
            existing_map = load_existing_file_shas(repo, branch)
        except Exception as e:
            if args.verbose:
                print(f"[incremental] failed to load existing file SHAs: {e}")
            existing_map = {}

    chunks: List[Tuple[str, int, int, str, str, str]] = []  # (path, start, end, text, file_sha, chunk_sha)
    changed_files = 0

    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        lines = read_lines(p)
        file_bytes = "\n".join(lines).encode("utf-8")
        file_sha = _sha1_bytes(file_bytes)

        key = (repo, branch, rel)
        if args.incremental and existing_map.get(key) == file_sha:
            # unchanged — skip
            continue

        changed_files += 1
        for (start, end, content) in chunk_lines(lines, args.max_lines, args.overlap):
            # hard cap per-chunk content length too (extra guard)
            if len(content) > args.request_max_chars:
                content = content[:args.request_max_chars]
            chunk_id = f"{repo}:{branch}:{rel}:{start}-{end}"
            chunk_sha = _sha1_bytes(chunk_id.encode("utf-8"))
            chunks.append((rel, start, end, content, file_sha, chunk_sha))

    skipped = total_files - changed_files
    print(f"Incremental mode: {skipped} files skipped (unchanged). {changed_files} files to (re)embed.")

    if not chunks:
        print("Nothing to embed. (No changed files or no eligible files)")
        return

    # Prepare embedding input pieces
    pieces = [c[3] for c in chunks]
    batches = pack_batches(
        pieces,
        piece_max_tokens=args.piece_max_tokens,
        request_max_tokens=args.request_max_tokens,
        request_max_chars=args.request_max_chars,
        batch_size=args.batch_size,
    )
    total_batches = len(batches)
    total_pieces = sum(len(b) for b in batches)
    if args.verbose:
        print(f"Prepared {total_pieces} pieces across {total_batches} batches.")
        print(f"Model={EMBED_MODEL} dims={EMBED_DIMS} — request_max_tokens={args.request_max_tokens}, "
              f"piece_max_tokens={args.piece_max_tokens}, request_max_chars={args.request_max_chars}, "
              f"batch_size={args.batch_size}")

    if args.dry_run:
        print(f"[DRY RUN] would embed {len(chunks)} chunks across {changed_files} files for {repo}@{branch}")
        return

    # Embed & upsert
    out_rows: List[dict] = []
    t0 = time.time()
    meta = chunks  # same order as pieces

    batch_start_idx = 0
    for bi, batch_inputs in enumerate(batches, start=1):
        batch_len = len(batch_inputs)
        batch_end_idx = batch_start_idx + batch_len
        meta_slice = meta[batch_start_idx:batch_end_idx]

        est_tokens = sum(rough_token_estimate(s) for s in batch_inputs)
        est_chars = sum(len(s) for s in batch_inputs)
        if args.verbose:
            print(f"[embed {bi}/{total_batches}] pieces={batch_len} est_tokens≈{est_tokens} est_chars={est_chars} …", end="", flush=True)

        bt0 = time.time()
        try:
            embeddings = embed_batch_safe(batch_inputs)
        except Exception as e:
            if args.verbose:
                print("\r", end="")
            print(f"[embed {bi}/{total_batches}] ERROR: {e}")
            print("→ Hint: lower --batch-size / --request-max-* or --piece-max-tokens.")
            batch_start_idx = batch_end_idx
            continue
        bt1 = time.time()

        if len(embeddings) != batch_len:
            if args.verbose:
                print("\r", end="")
            print(f"[embed {bi}/{total_batches}] ERROR: returned {len(embeddings)} embeddings for {batch_len} inputs")
            batch_start_idx = batch_end_idx
            continue

        rows: List[dict] = []
        for k in range(batch_len):
            rel, start, end, content, file_sha, chunk_sha = meta_slice[k]
            rows.append({
                "repo": repo,
                "branch": branch,
                "path": rel,
                "start_line": start,
                "end_line": end,
                "content": content,
                "file_sha": file_sha,
                "chunk_sha": chunk_sha,
                "commit_sha": os.getenv("GIT_COMMIT", "HEAD"),
                "embedding": embeddings[k],
                "dims": EMBED_DIMS,
            })

        try:
            sup_t0 = time.time()
            supabase_upsert_rows(rows, verbose=args.verbose)
            sup_t1 = time.time()
        except Exception as e:
            if args.verbose:
                print("\r", end="")
            print(f"[embed {bi}/{total_batches}] upsert ERROR: {e}")
            batch_start_idx = batch_end_idx
            continue

        out_rows.extend(rows)
        dt = bt1 - bt0
        dtsup = sup_t1 - sup_t0
        if args.verbose:
            print("\r", end="")
            rate = batch_len / max(0.001, dt)
            print(f"[embed {bi}/{total_batches}] ok {batch_len} pcs in {dt:.2f}s (rate {rate:.2f} pcs/s), supabase {dtsup:.2f}s")

        batch_start_idx = batch_end_idx

    t1 = time.time()
    print(f"Indexed {len(out_rows)} chunks across {len(set([r['path'] for r in out_rows]))} files @ {repo}@{branch}")
    print(f"Total time {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
