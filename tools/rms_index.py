#!/usr/bin/env python3
"""
rms_index.py — Repo Memory Service indexer (with detailed progress logging)

v2 — 2025-08-12
- Clean reset: no _hard_split_piece / split_by_chars_preserve_lines paths
- Robust batching, per-piece truncation, retries, verbose progress
- Works with SUPABASE_URL + SUPABASE_SERVICE_ROLE (or SUPABASE_KEY) and OPENAI_API_KEY

Usage (example):
  python tools/rms_index.py ^
    --repo-name personal-agent-ste ^
    --branch main ^
    --root . ^
    --incremental ^
    --paths "backend frontend/src frontend/vite.config.js frontend/tsconfig.json" ^
    --include-ext ".py,.ts,.tsx,.js,.jsx,.vue,.sql,.json,.md" ^
    --max-lines 80 --overlap 10 --max-chars 3000 ^
    --batch-size 4 --request-max-chars 10000 --request-max-tokens 6500 ^
    --piece-max-tokens 6000 --verbose
"""

from __future__ import annotations
import os, sys, json, time, math, argparse, hashlib, pathlib, signal
from typing import List, Tuple, Dict, Iterable, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from dotenv import load_dotenv

load_dotenv()  # load .env from repo root

VERSION = "rms-index v2 (2025-08-12)"

# ------------------------- Env & models -------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
EMBED_DIMS = 3072 if "3-large" in EMBED_MODEL else 1536
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print(
        "Missing SUPABASE_URL and/or SUPABASE_SERVICE_ROLE (or SUPABASE_KEY).",
        file=sys.stderr,
    )
    sys.exit(1)
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY.", file=sys.stderr)
    sys.exit(1)


# ------------------------- HTTP helpers -------------------------
def _rest(
    url: str,
    method: str = "GET",
    body: Optional[dict] = None,
    supabase_key: Optional[str] = None,
) -> dict:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Prefer", "resolution=merge-duplicates")
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


# ------------------------- OpenAI embeddings -------------------------
def openai_embeddings_create(
    model: str, inputs: List[str], timeout: int = 90, max_retries: int = 4
) -> List[List[float]]:
    assert inputs, "inputs must be non-empty"
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    payload = {"model": model, "input": inputs}
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
            for k, v in headers.items():
                req.add_header(k, v)
            with urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                data = raw.get("data") or []
                return [d["embedding"] for d in data]
        except HTTPError as e:
            msg = ""
            try:
                msg = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                print(
                    f"[emb] HTTP {e.code}; retrying in {backoff:.1f}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(backoff)
                backoff *= 1.8
                continue
            raise RuntimeError(f"OpenAI error {e.code}: {msg}") from None
        except URLError as e:
            if attempt < max_retries:
                print(
                    f"[emb] network error; retrying in {backoff:.1f}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(backoff)
                backoff *= 1.8
                continue
            raise RuntimeError(f"OpenAI network error: {e}") from None
    raise RuntimeError("unreachable")


# ------------------------- Chunking -------------------------
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", ".next", ".turbo"}
DEFAULT_EXTS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".vue",
    ".sql",
    ".json",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".env",
    ".txt",
}


def file_iter(
    root: pathlib.Path, include_ext: set[str], paths: Optional[List[str]]
) -> Iterable[pathlib.Path]:
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
            if allow_prefixes and not any(
                str(p).startswith(str(ap)) for ap in allow_prefixes
            ):
                continue
            yield p


def read_lines(path: pathlib.Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []


def chunk_lines(
    lines: List[str], max_lines: int = 80, overlap: int = 10
) -> Iterable[Tuple[int, int, str]]:
    n = len(lines)
    if n == 0:
        return
    i = 0
    while i < n:
        j = min(n, i + max_lines)
        start, end = i + 1, j  # 1-based
        text = "\n".join(lines[i:j])
        yield start, end, text
        if j >= n:
            break
        i = j - overlap if overlap > 0 else j


def sha1_bytes(b: bytes) -> str:
    import hashlib

    return hashlib.sha1(b).hexdigest()


# ------------------------- Supabase upsert -------------------------
def supabase_upsert_rows(rows: list[dict], verbose: bool = False):
    if not rows:
        return

    url_chunks = f"{SUPABASE_URL}/rest/v1/repo_chunks?on_conflict=chunk_sha"
    url_memory = f"{SUPABASE_URL}/rest/v1/repo_memory?on_conflict=ux_repo_memory_chunk"

    def _send(url, payload):
        try:
            _rest(url, method="POST", body=payload, supabase_key=SUPABASE_KEY)
        except RuntimeError as e:
            if ("on_conflict" in url) and (
                "does not exist" in str(e).lower()
                or "42703" in str(e)
                or "PGRST204" in str(e)
            ):
                fallback = url.split("?")[0]
                if verbose:
                    print(f"[db] falling back to plain insert: {fallback}")
                _rest(fallback, method="POST", body=payload, supabase_key=SUPABASE_KEY)
            else:
                raise

    # ---------- repo_chunks ----------
    chunk_rows = [
        {
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
            # include if your table has it; otherwise omit:
            "chunk_index": r.get("chunk_index"),
        }
        for r in rows
    ]

    # ---------- repo_memory ----------
    # Your table columns (from your screenshots):
    # id, repo_name, repo, branch, head_ref, path, file_sha, chunk_sha,
    # start_line, end_line, content, embedding, created_at, updated_at,
    # dims, commit_sha
    mem_rows_raw = [
        {
            "repo_name": r["repo"],  # required in your table
            "repo": r["repo"],  # you added this column
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
            # DO NOT send: "lang", "meta", "framework", "service"
        }
        for r in rows
    ]

    # Drop keys with None to avoid NOT NULL issues / schema cache errors
    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    mem_rows = [_clean(m) for m in mem_rows_raw]

    _send(url_chunks, chunk_rows)
    _send(url_memory, mem_rows)


# ------------------------- Token/size helpers -------------------------
def rough_token_estimate(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))


def split_into_batches(
    texts: List[str],
    piece_max_tokens: int,
    request_max_tokens: int,
    batch_size: int,
    request_max_chars: int,
) -> List[List[str]]:
    """Truncate each piece to piece_max_tokens*~4 chars; then greedy-pack into batches."""
    safe: List[str] = []
    per_piece_char_cap = max(100, piece_max_tokens * 4)
    for s in texts:
        if not isinstance(s, str):
            s = (
                ""
                if s is None
                else (json.dumps(s) if isinstance(s, (dict, list)) else str(s))
            )
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
            if ((cur_tokens + t) <= request_max_tokens) and (
                (cur_chars + len(s)) <= request_max_chars
            ):
                cur.append(s)
                cur_tokens += t
                cur_chars += len(s)
                i += 1
            else:
                if not cur:
                    cur.append(s)
                    i += 1
                break
        batches.append(cur)
    return batches


# ------------------------- Main -------------------------
STOP = False


def _handle_sigint(sig, frame):
    global STOP
    STOP = True
    print("\n[!] Received interrupt — will stop after current batch.")


signal.signal(signal.SIGINT, _handle_sigint)


def main():
    print(f"{VERSION} — model={EMBED_MODEL} dims={EMBED_DIMS}")
    ap = argparse.ArgumentParser(
        description="Index repo into Supabase with embeddings (progress logging)."
    )
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--paths",
        default="",
        help='Quoted space-separated list, e.g. "backend frontend/src file.js"',
    )
    ap.add_argument("--include-ext", default=",".join(sorted(DEFAULT_EXTS)))
    ap.add_argument("--max-lines", type=int, default=80)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Max chars per chunk text before piece truncation",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--request-max-chars", type=int, default=10000)
    ap.add_argument("--request-max-tokens", type=int, default=6500)
    ap.add_argument("--piece-max-tokens", type=int, default=6000)
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Skip unchanged files based on file SHA",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = args.repo_name
    branch = args.branch
    root = pathlib.Path(args.root).resolve()
    include_ext = {e.strip().lower() for e in args.include_ext.split(",") if e.strip()}
    paths = [p for p in args.paths.split() if p.strip()] if args.paths else None

    files = list(file_iter(root, include_ext, paths))
    total_files = len(files)

    # Incremental: map (repo,branch,path) -> file_sha from DB
    existing_map: Dict[Tuple[str, str, str], str] = {}
    if args.incremental:
        try:
            url = f"{SUPABASE_URL}/rest/v1/repo_chunks?select=repo,branch,path,file_sha"
            rows = _rest(url, method="GET", supabase_key=SUPABASE_KEY)
            for r in rows:
                key = (r.get("repo"), r.get("branch"), r.get("path"))
                if key[0] and key[1] and key[2] and r.get("file_sha"):
                    existing_map[key] = r["file_sha"]
        except Exception:
            existing_map = {}

    chunks: List[Tuple[str, int, int, str, str, str]] = (
        []
    )  # (path, start, end, text, file_sha, chunk_sha)
    changed_files = 0
    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        lines = read_lines(p)
        file_bytes = "\n".join(lines).encode("utf-8")
        file_sha = sha1_bytes(file_bytes)
        key = (repo, branch, rel)
        if args.incremental and existing_map.get(key) == file_sha:
            continue
        changed_files += 1
        for start, end, content in chunk_lines(lines, args.max_lines, args.overlap):
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            if len(content) > args.max_chars:
                content = content[: args.max_chars]
            chunk_id = f"{repo}:{branch}:{rel}:{start}-{end}"
            chunk_sha = sha1_bytes(chunk_id.encode("utf-8"))
            chunks.append((rel, start, end, content, file_sha, chunk_sha))

    skipped = total_files - changed_files
    print(
        f"Incremental mode: {skipped} files skipped (unchanged). {changed_files} files to (re)embed."
    )

    pieces = [c[3] for c in chunks]
    if not pieces:
        print("Nothing to embed. (No changed files or no eligible files)")
        return

    batches = split_into_batches(
        pieces,
        piece_max_tokens=args.piece_max_tokens,
        request_max_tokens=args.request_max_tokens,
        batch_size=args.batch_size,
        request_max_chars=args.request_max_chars,
    )

    total_batches = len(batches)
    total_pieces = sum(len(b) for b in batches)
    if args.verbose:
        print(f"Prepared {total_pieces} pieces across {total_batches} batches.")
        print(
            f"Model={EMBED_MODEL} dims={EMBED_DIMS} — request_max_tokens={args.request_max_tokens}, "
            f"piece_max_tokens={args.piece_max_tokens}, request_max_chars={args.request_max_chars}, "
            f"batch_size={args.batch_size}"
        )

    if args.dry_run:
        print(
            f"DRY RUN — would embed {len(chunks)} chunks across {changed_files} files for {repo}@{branch}"
        )
        return

    out_rows: List[dict] = []
    t0 = time.time()
    meta = chunks  # same order as pieces

    batch_start_idx = 0
    for bi, batch_inputs in enumerate(batches, start=1):
        if STOP:
            print("[!] Stopping early by user request.")
            break

        batch_len = len(batch_inputs)
        batch_end_idx = batch_start_idx + batch_len
        meta_slice = meta[batch_start_idx:batch_end_idx]

        est_tokens = sum(rough_token_estimate(s) for s in batch_inputs)
        est_chars = sum(len(s) for s in batch_inputs)
        print(
            f"[embed {bi}/{total_batches}] pieces={batch_len} est_tokens≈{est_tokens} est_chars={est_chars} …",
            end="",
            flush=True,
        )

        bt0 = time.time()
        try:
            embeddings = openai_embeddings_create(EMBED_MODEL, batch_inputs)
        except Exception as e:
            print("\r", end="")
            print(f"[embed {bi}/{total_batches}] ERROR: {e}")
            print("→ Hint: try lowering --batch-size or --request-max-*.")
            batch_start_idx = batch_end_idx
            continue
        bt1 = time.time()

        if len(embeddings) != batch_len:
            print("\r", end="")
            print(
                f"[embed {bi}/{total_batches}] ERROR: returned {len(embeddings)} embeddings for {batch_len} inputs"
            )
            batch_start_idx = batch_end_idx
            continue

        rows: List[dict] = []
        for k in range(batch_len):
            rel, start, end, content, file_sha, chunk_sha = meta_slice[k]
            rows.append(
                {
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
                }
            )

        try:
            sup_t0 = time.time()
            supabase_upsert_rows(rows, verbose=args.verbose)
            sup_t1 = time.time()
        except Exception as e:
            print("\r", end="")
            print(f"[embed {bi}/{total_batches}] upsert ERROR: {e}")
            batch_start_idx = batch_end_idx
            continue

        out_rows.extend(rows)
        dt = bt1 - bt0
        dtsup = sup_t1 - sup_t0
        print("\r", end="")
        rate = batch_len / max(0.001, dt)
        print(
            f"[embed {bi}/{total_batches}] ok in {dt:.2f}s (rate {rate:.2f} pcs/s), supabase {dtsup:.2f}s"
        )

        batch_start_idx = batch_end_idx

    t1 = time.time()
    print(
        f"Indexed {len(out_rows)} chunks across {len(set([r['path'] for r in out_rows]))} files @ {repo}@{branch}"
    )
    print(f"Total time {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
