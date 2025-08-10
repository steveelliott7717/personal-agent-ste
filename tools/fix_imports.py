# tools/fix_imports.py
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
BACKEND = ROOT / "backend"

# packages we want to prefix with "backend."
PKGS = r"(agents|services|utils|plugins|semantics|workers|reasoner|scripts|config)"

# Patterns:
#   from services.x import y   -> from backend.services.x import y
#   from semantics import z    -> from backend.semantics import z
#   import utils               -> import backend.utils as utils
FROM_RX = re.compile(rf"(^\s*from\s+){PKGS}(\s*\.)", re.MULTILINE)
FROM_BARE_RX = re.compile(rf"(^\s*from\s+){PKGS}(\s+import\s+)", re.MULTILINE)
IMPORT_RX = re.compile(rf"(^\s*)import\s+{PKGS}(\b)", re.MULTILINE)

def patch_text(text: str) -> str:
    text = FROM_RX.sub(lambda m: f"{m.group(1)}backend.{m.group(2)[0:-1]}.", text)
    text = FROM_BARE_RX.sub(lambda m: f"{m.group(1)}backend.{m.group(2).strip().split()[0]} ", text)
    # import pkg  -> import backend.pkg as pkg
    def imp_sub(m):
        pkg = m.group(2)
        return f"{m.group(1)}import backend.{pkg} as {pkg}"
    text = IMPORT_RX.sub(imp_sub, text)
    return text

def main():
    files = list(BACKEND.rglob("*.py"))
    changed = 0
    for f in files:
        s = f.read_text(encoding="utf-8")
        p = patch_text(s)
        if p != s:
            f.write_text(p, encoding="utf-8")
            changed += 1
            print(f"[fix] {f.relative_to(ROOT)}")
    print(f"Done. Patched {changed} files.")

if __name__ == "__main__":
    main()
