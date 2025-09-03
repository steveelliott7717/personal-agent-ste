# tools/find_unused_frontend.py
"""
Scan the frontend for .vue/.js files that are probably unused:
- Builds a reference graph from import statements and router files
- Roots: src/main.js and any router index.js
- Reports files never reached and not referenced by index.html
Limitations: naive parser (regex), but good enough to flag likely dead files.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict, deque

FRONTEND = Path("frontend")
SRC = FRONTEND / "src"
PUBLIC = FRONTEND / "public"

# Roots (entry points)
ROOTS = [
    SRC / "main.js",
    SRC / "router" / "index.js",
    SRC / "router.js",  # if it exists
]

IMPORT_RE = re.compile(
    r"""(?:import\s+(?:.+?\s+from\s+)?|import\()\s*     # import ... from OR dynamic import(
        ['"](.+?)['"]                                   # 'path'
    """,
    re.VERBOSE,
)


# Resolve an import path to a real file (basic heuristics)
def resolve_import(base_file: Path, import_path: str) -> Path | None:
    if import_path.startswith(("http://", "https://")):
        return None
    if import_path.startswith("@/"):
        rel = SRC / import_path[2:]
    elif import_path.startswith("/"):
        # treat as from project root public/src
        rel = SRC.parent / import_path.lstrip("/")
    else:
        rel = (base_file.parent / import_path).resolve()

    # Try as-is
    candidates = [rel]
    # Try with extensions
    if rel.suffix == "":
        candidates += [
            rel.with_suffix(".js"),
            rel.with_suffix(".vue"),
            rel / "index.js",
            rel / "index.vue",
        ]

    for c in candidates:
        if c.exists() and SRC in c.parents:
            return c
    return None


def find_all_source_files():
    files = set()
    for ext in ("*.js", "*.vue"):
        files.update(SRC.rglob(ext))
    return files


def build_graph(files):
    graph = defaultdict(set)
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in IMPORT_RE.finditer(text):
            imp = m.group(1)
            tgt = resolve_import(f, imp)
            if tgt:
                graph[f].add(tgt)
    return graph


def reachable(graph, roots):
    seen = set()
    q = deque()
    for r in roots:
        if r.exists():
            q.append(r)
            seen.add(r)
    while q:
        u = q.popleft()
        for v in graph.get(u, ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def index_html_refs():
    refs = set()
    idx = FRONTEND / "index.html"
    if idx.exists():
        txt = idx.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'href=["\'](/.+?)["\']|src=["\'](/.+?)["\']', txt):
            path = (FRONTEND / (m.group(1) or m.group(2)).lstrip("/")).resolve()
            if path.exists():
                refs.add(path)
    return refs


def main():
    files = find_all_source_files()
    graph = build_graph(files)
    roots = [r for r in ROOTS if r.exists()]
    seen = reachable(graph, roots)
    html_refs = index_html_refs()

    # Candidates: in src but not reachable from roots and not referenced by index.html
    unused = sorted(f for f in files if f not in seen and f not in html_refs)

    print("== Frontend unused candidates ==")
    if not unused:
        print("(none)")
        return

    for f in unused:
        print(f" - {f.as_posix()}")

    print("\nTip: review and then remove with:")
    print("git rm -f \\")
    for i, f in enumerate(unused):
        sep = " ^" if i < len(unused) - 1 else ""
        print(f"  {f.as_posix()}{sep}")


if __name__ == "__main__":
    sys.exit(main())
