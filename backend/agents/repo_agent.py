# backend/agents/repo_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI
from backend.services.rms import repo_search, format_citation

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-5")

SYSTEM = (
"ROLE\n"
"You are a senior engineer operating inside {REPO}@{BRANCH}@{COMMIT}.\n"
"You MUST rely ONLY on the RMS context provided here; do not invent unseen code.\n"
"Cite every code reference as [n] path:start–end@sha.\n\n"
"MISSION\n"
"Complete the TASK below with minimal, safe changes. If context is missing, STOP and list files/lines to recall.\n\n"
"RESPONSE SHAPE\n"
"1) TL;DR\n2) PLAN\n3) PATCH(ES) — unified diff with [n] citations\n4) TESTS\n5) NOTES\n"
)

def _embed(text: str) -> List[float]:
    client = OpenAI()
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding

def propose_changes(task: str, *, repo: str, branch: str, commit: str = "HEAD",
                    k: int = 12, path_prefix: Optional[str] = None) -> Dict[str, Any]:
    vec = _embed(task)
    hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []
    # build prompt with citations + content
    ctx = []
    for i, h in enumerate(hits, start=1):
        cite = format_citation(i, h["path"], h["start_line"], h["end_line"], h["commit_sha"])
        ctx.append(f"{cite}\n```\n{h['content']}\n```")
    prompt = (
        SYSTEM.format(REPO=repo, BRANCH=branch, COMMIT=commit)
        + "\nTASK\n" + task
        + "\n\nCONTEXT\n" + ("\n\n".join(ctx) if ctx else "(no context)")
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Follow RESPONSE SHAPE exactly; produce minimal, safe diffs."},
            {"role": "user", "content": prompt},
        ]
    )
    return {"hits": hits, "draft": resp.choices[0].message.content, "prompt": prompt}
