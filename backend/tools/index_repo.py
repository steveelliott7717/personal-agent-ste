# tools/index_repo.py
import os, pathlib, re
from typing import List, Tuple
from supabase import create_client
from openai import OpenAI

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ[
    "SUPABASE_SERVICE_ROLE"
]  # service role recommended for writes
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

REPO = "personal-agent-ste"
BRANCH = "main"


# Very simple chunker (line-based)
def chunk_file(
    text: str, width: int = 1200, overlap: int = 200
) -> List[Tuple[int, int, str]]:
    lines = text.splitlines()
    chunks = []
    buf, start = [], 1
    length = 0
    for i, line in enumerate(lines, start=1):
        length += len(line) + 1
        buf.append(line)
        if length >= width:
            chunks.append((start, i, "\n".join(buf)))
            # overlap reset
            back = max(0, len(buf) - overlap)
            keep = buf[back:]
            start = i - len(keep) + 1
            buf = keep[:]
            length = sum(len(x) + 1 for x in buf)
    if buf:
        chunks.append((start, len(lines), "\n".join(buf)))
    return chunks


def should_index(path: str) -> bool:
    return bool(re.search(r"\.(py|js|ts|vue|md|json|yml|yaml|toml|sql)$", path))


ROOT = os.environ.get("RMS_ROOT", "/app")
# and later use ROOT in file walk logic

paths = [p for p in ROOT.rglob("*") if p.is_file() and should_index(str(p))]
print(f"Indexing {len(paths)} files...")

for p in paths:
    txt = p.read_text(errors="ignore")
    for start, end, chunk in chunk_file(txt):
        emb = (
            client.embeddings.create(model="text-embedding-3-small", input=chunk)
            .data[0]
            .embedding
        )
        row = {
            "repo": REPO,
            "branch": BRANCH,
            "path": str(p).replace("\\", "/"),
            "start_line": start,
            "end_line": end,
            "content": chunk,
            "embedding": emb,
            "commit_sha": "HEAD",
        }
        sb.table("repo_chunks").insert(row).execute()

print("Done.")
