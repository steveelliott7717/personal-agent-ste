# backend/tools/index_repo_ci.py
from __future__ import annotations

import os, sys, json, time, math, argparse, pathlib, hashlib, signal, re
from typing import List, Tuple, Dict, Iterable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- OpenAI client (1.x) --------
try:
    from openai import OpenAI
except Exception as e:
    print("Missing openai>=1.x. pip install --upgrade openai", file=sys.stderr)
    raise

# -------- Optional tokenizer (tiktoken) --------
try:
    import tiktoken
except Exception:
    tiktoken = None

# =========================================
#               CONFIG
# =========================================
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
EMBED_DIMS  = 3072 if "3-large" in EMBED_MODEL else 1536
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Missing SUPABASE_URL and/or SUPABASE_SERVICE_ROLE (or SUPABASE_KEY).", file=sys.stderr)
    sys.exit(1)
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY.", file=sys.stderr)
    sys.exit(1)

# Safe per-item token cap (well below 8192)
PIECE_MAX_TOKENS_DEFAULT = int(os.getenv("PIECE_MAX_TOKENS", "5200"))


# Batch controls; embeddings endpoint doesn’t enforce total tokens across items,
# but we keep batches reasonable for reliability.
BATCH_SIZE_DEFAULT       = int(os.getenv("BATCH_SIZE", "8"))

# File filters
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", ".next", ".turbo", ".cache"}
DEFAULT_EXTS = {
    ".py",".ts",".tsx",".js",".jsx",".vue",".sql",".json",".md",".yml",".yaml",
    ".toml",".ini",".env",".txt",".css",".html",".sh"
}

STOP = False
def _handle_sigint(sig, frame):
    global STOP
    STOP = True
    print("\n[!] Received interrupt — will stop after current batch.", flush=True)
signal.signal(signal.SIGINT, _handle_sigint)

# =========================================
#           Helpers: HTTP (Supabase)
# =========================================
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ---- Token + batching helpers ----
def rough_token_estimate(s: str) -> int:
    # Very safe overestimate: ~3.3 chars/token; clamp >= 1
    return max(1, int(len(s) / 3.3))

def pack_batches(items, *, request_max_tokens: int, batch_size: int, request_max_chars: int | None = None):
    """
    Greedy-pack strings into batches so the **sum** of estimated tokens
    (and optional chars) stays <= request_max_* limits.
    """
    batches, cur, cur_tok, cur_chars = [], [], 0, 0
    for s in items:
        t = rough_token_estimate(s)
        c = len(s)
        # If adding this item would exceed any cap, flush current batch first
        would_exceed_tokens = (cur_tok + t) > request_max_tokens
        would_exceed_chars  = request_max_chars is not None and (cur_chars + c) > request_max_chars
        would_exceed_len    = len(cur) >= batch_size
        if cur and (would_exceed_tokens or would_exceed_chars or would_exceed_len):
            batches.append(cur)
            cur, cur_tok, cur_chars = [], 0, 0
        # If it's still too big for an empty batch, force it alone; upper layers will handle retry
        cur.append(s); cur_tok += t; cur_chars += c
    if cur:
        batches.append(cur)
    return batches


def _rest(url: str, method: str = "GET", body: Optional[dict] = None, supabase_key: Optional[str] = None) -> dict:
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

# =========================================
#           Tokenization utilities
# =========================================
_enc_cache = None

def _get_encoder():
    global _enc_cache
    if _enc_cache is not None:
        return _enc_cache
    if tiktoken is None:
        _enc_cache = None
        return None
    try:
        _enc_cache = tiktoken.encoding_for_model(EMBED_MODEL)
    except Exception:
        # Fallback to cl100k if model not known
        try:
            _enc_cache = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc_cache = None
    return _enc_cache

def token_len(s: str) -> int:
    enc = _get_encoder()
    if enc:
        try:
            return len(enc.encode(s))
        except Exception:
            pass
    # Conservative fallback (≈3 chars/token to avoid underestimation)
    return max(1, math.ceil(len(s) / 3))

def hard_truncate_piece(s: str, piece_max_tokens: int, absolute_char_cap: int = 12000) -> str:
    # Cap by both tokens (~chars) and a strict char ceiling (more conservative)
    per_piece_char_cap = min(piece_max_tokens * 3, absolute_char_cap)
    return s if len(s) <= per_piece_char_cap else s[:per_piece_char_cap]

def split_long_text(text: str, max_tokens: int) -> Iterable[str]:
    """
    Split text so each piece <= max_tokens. Uses tiktoken when available; otherwise
    falls back to size-based windows with paragraph/line boundaries.
    """
    enc = _get_encoder()
    if enc:
        toks = enc.encode(text)
        n = len(toks)
        for i in range(0, n, max_tokens):
            yield enc.decode(toks[i:i+max_tokens])
        return

    # Fallback path without tiktoken: chunk by ~chars with soft boundaries
    approx_chars = max_tokens * 3  # safer than *4
    if len(text) <= approx_chars:
        yield text
        return

    # Prefer splitting on double newlines, then lines, else fixed window
    parts = re.split(r"(\n\s*\n)", text)  # keep separators implicit by re-joining
    buf = []
    buf_len = 0
    for part in parts:
        chunk = part
        if buf_len + len(chunk) <= approx_chars:
            buf.append(chunk); buf_len += len(chunk)
        else:
            if buf:
                yield "".join(buf)
                buf, buf_len = [chunk], len(chunk)
            else:
                # Single part larger than window -> hard slice
                s = chunk
                step = approx_chars
                for i in range(0, len(s), step):
                    yield s[i:i+step]
                buf, buf_len = [], 0
    if buf:
        yield "".join(buf)


def ensure_under_limit(text: str, max_tokens: int) -> List[str]:
    if token_len(text) <= max_tokens:
        return [text]
    return list(split_long_text(text, max_tokens))

# =========================================
#           Repo scanning / chunking
# =========================================
def sha1_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()

def read_file_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def file_iter(root: pathlib.Path, include_ext: set[str], paths: Optional[List[str]]) -> Iterable[pathlib.Path]:
    base = root.resolve()
    allow_prefixes = []
    if paths:
        for q in paths:
            allow_prefixes.append((base / q).resolve())
    for p in base.rglob("*"):
        if p.is_dir():
            if p.name in SKIP_DIRS:
                continue
        elif p.is_file():
            if include_ext and p.suffix.lower() not in include_ext:
                continue
            if allow_prefixes and not any(str(p).startswith(str(ap)) for ap in allow_prefixes):
                continue
            yield p

def logical_chunks_by_lines(text: str, max_lines: int, overlap: int) -> Iterable[str]:
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return
    i = 0
    while i < n:
        j = min(n, i + max_lines)
        piece = "\n".join(lines[i:j])
        yield piece
        if j >= n: break
        i = j - overlap if overlap > 0 else j

# =========================================
#            Supabase upsert
# =========================================
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

    # repo_chunks rows
    chunk_rows = [{
        "repo": r["repo"],
        "branch": r["branch"],
        "path": r["path"],
        "start_line": r.get("start_line"),
        "end_line": r.get("end_line"),
        "content": r["content"],
        "file_sha": r.get("file_sha"),
        "chunk_sha": r.get("chunk_sha"),
        "commit_sha": r.get("commit_sha"),
        "head_ref": r.get("commit_sha") or "HEAD",
        "chunk_index": r.get("chunk_index"),
    } for r in rows]

    # repo_memory rows (embedding table)
    mem_rows_raw = [{
        "repo_name": r["repo"],
        "repo": r["repo"],
        "branch": r["branch"],
        "head_ref": r.get("commit_sha") or "HEAD",
        "path": r["path"],
        "file_sha": r.get("file_sha"),
        "chunk_sha": r.get("chunk_sha"),
        "start_line": r.get("start_line"),
        "end_line": r.get("end_line"),
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

# =========================================
#            Embedding + pipeline
# =========================================
client = OpenAI(api_key=OPENAI_API_KEY)

def embed_batch(inputs: List[str]) -> List[List[float]]:
    """Embed a batch of strings. Assumes each item is already <= piece token cap."""
    assert inputs, "embed_batch: empty inputs"
    # Defensive: drop empty strings (OpenAI rejects some empty inputs)
    safe = [s if (s and s.strip()) else " " for s in inputs]
    resp = client.embeddings.create(model=EMBED_MODEL, input=safe)
    data = resp.data or []
    return [d.embedding for d in data]

def build_rows_for_file(
    *,
    repo: str,
    branch: str,
    rel_path: str,
    file_text: str,
    file_sha: str,
    piece_max_tokens: int,
    max_lines: int,
    overlap: int,
) -> List[Tuple[str, int, int, str]]:
    """
    Return list of (path, start_line, end_line, content) with per-piece token safety.
    We first chunk by lines (for code locality), then split any oversized piece by tokens.
    """
    rows: List[Tuple[str,int,int,str]] = []
    # First pass: logical line chunks
    start_line = 1
    for piece in logical_chunks_by_lines(file_text, max_lines=max_lines, overlap=overlap):
        end_line = start_line + piece.count("\n")
        # Enforce token cap for each piece
        subpieces = ensure_under_limit(piece, piece_max_tokens)
        if len(subpieces) == 1:
            rows.append((rel_path, start_line, end_line, subpieces[0]))
        else:
            # If we split, approximate sub-line ranges (best-effort)
            approx_span = max(1, (end_line - start_line + 1) // len(subpieces))
            s = start_line
            for sp in subpieces:
                e = s + approx_span - 1
                rows.append((rel_path, s, e, sp))
                s = e + 1
        start_line = end_line + 1
    return rows

def main():
    ap = argparse.ArgumentParser(description="Index repo into Supabase with embeddings (token-safe).")
    ap.add_argument("--repo-name", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--paths", default="", help='Quoted space-separated list, e.g. "backend frontend/src file.js"')
    ap.add_argument("--include-ext", default=",".join(sorted(DEFAULT_EXTS)))
    ap.add_argument("--max-lines", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--piece-max-tokens", type=int, default=PIECE_MAX_TOKENS_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = args.repo_name
    branch = args.branch
    root = pathlib.Path(args.root).resolve()
    include_ext = {e.strip().lower() for e in args.include_ext.split(",") if e.strip()}
    paths = [p for p in args.paths.split() if p.strip()] if args.paths else None

    files = list(file_iter(root, include_ext, paths))
    if args.verbose:
        print(f"[indexer] Scanning {len(files)} files...")

    all_chunks: List[Tuple[str,int,int,str,str,str]] = []  # (path, start, end, content, file_sha, chunk_sha)

    for p in files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        text = read_file_text(p)
        file_sha = sha1_bytes(text.encode("utf-8"))
        line_chunks = build_rows_for_file(
            repo=repo,
            branch=branch,
            rel_path=rel,
            file_text=text,
            file_sha=file_sha,
            piece_max_tokens=args.piece_max_tokens,
            max_lines=args.max_lines,
            overlap=args.overlap,
        )
        count_before = len(line_chunks)
        # Pack and compute chunk_sha
        idx = 0
        for (path_rel, start, end, content) in line_chunks:
            chunk_id = f"{repo}:{branch}:{path_rel}:{start}-{end}:{idx}"
            chunk_sha = sha1_bytes(chunk_id.encode("utf-8"))
            all_chunks.append((path_rel, start, end, content, file_sha, chunk_sha))
            idx += 1

        print(f"[indexer] OK {rel} ({count_before} chunks)")

        if STOP:
            break

    if not all_chunks:
        print("[indexer] No chunks to embed. Done.")
        return

    if args.dry_run:
        print(f"[indexer] DRY RUN — would embed {len(all_chunks)} chunks across {len(set([c[0] for c in all_chunks]))} files.")
        return

    # Embed & Upsert in batches
    total = len(all_chunks)
    batch_size = max(1, int(args.batch_size))
    i = 0
    out_rows: List[dict] = []

    while i < total and not STOP:
        j = min(total, i + batch_size)
        meta_slice = all_chunks[i:j]
        inputs = [m[3] for m in meta_slice]

        # Final safety: if any input accidentally exceeds cap, split it locally
        # and process as a sub-batch to avoid 400s.
        sub_inputs: List[str] = []
        sub_meta: List[Tuple[str,int,int,str,str,str]] = []
        for m in meta_slice:
            base_path, sline, eline, text, fsha, csha = m

            # First, split anything over the configured cap
            if token_len(text) > args.piece_max_tokens:
                parts = ensure_under_limit(text, args.piece_max_tokens)
            else:
                parts = [text]

            # Extra hard guard: if any part still exceeds ~cap (due to estimator),
            # iteratively shrink by chars until safely under.
            safe_parts: List[str] = []
            for part in parts:
                # quick char clamp to avoid pathological strings
                part = hard_truncate_piece(part, args.piece_max_tokens)
                # iterative tighten if needed
                attempts = 0
                while token_len(part) > args.piece_max_tokens and attempts < 6:
                    # shrink by 10% and try again
                    part = part[: max(1, int(len(part) * 0.9))]
                    attempts += 1
                safe_parts.append(part)

            # Append with sub-index if we split
            if len(safe_parts) == 1:
                sub_inputs.append(safe_parts[0])
                sub_meta.append((base_path, sline, eline, safe_parts[0], fsha, csha))
            else:
                for sub_idx, sp in enumerate(safe_parts):
                    sub_inputs.append(sp)
                    sub_meta.append((base_path, sline, eline, sp, fsha, f"{csha}-{sub_idx}"))


        try:
            t0 = time.time()
            vecs = embed_batch(sub_inputs)
            t1 = time.time()
        except Exception as e:
            print(f"[embed {i//batch_size+1}] ERROR: {e}", flush=True)
            # Skip bad batch but continue; next runs will still try others
            i = j
            continue

        if len(vecs) != len(sub_meta):
            print(f"[embed {i//batch_size+1}] ERROR: returned {len(vecs)} embeddings for {len(sub_meta)} inputs", flush=True)
            i = j
            continue

        rows: List[dict] = []
        for k, emb in enumerate(vecs):
            rel, start, end, content, file_sha, chunk_sha = sub_meta[k]
            rows.append({
                "repo": repo, "branch": branch, "path": rel,
                "start_line": start, "end_line": end,
                "content": content, "file_sha": file_sha, "chunk_sha": chunk_sha,
                "commit_sha": os.getenv("GIT_COMMIT", "HEAD"),
                "embedding": emb, "dims": EMBED_DIMS,
                "chunk_index": i + k,
            })

        try:
            sup_t0 = time.time()
            supabase_upsert_rows(rows, verbose=args.verbose)
            sup_t1 = time.time()
        except Exception as e:
            print(f"[embed {i//batch_size+1}] upsert ERROR: {e}", flush=True)
            i = j
            continue

        out_rows.extend(rows)
        rate = len(sub_inputs) / max(0.001, (t1 - t0))
        print(f"[embed {i//batch_size+1}] ok {len(sub_inputs)} pcs in {(t1 - t0):.2f}s (rate {rate:.2f} pcs/s), supabase {(sup_t1 - sup_t0):.2f}s", flush=True)
        i = j

    print(f"Indexed {len(out_rows)} chunks across {len(set([r['path'] for r in out_rows]))} files @ {repo}@{branch}")
    print("Done.")

if __name__ == "__main__":
    main()
