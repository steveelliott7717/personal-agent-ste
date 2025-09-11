# backend/repo/prompts.py
from __future__ import annotations
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _load_prompt(filename: str) -> str:
    """Loads a prompt file from the backend/prompts directory."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        # Fallback for different structures, e.g. when tests run from a different root
        path = Path(__file__).resolve().parents[2] / "backend" / "prompts" / filename
        if not path.exists():
            return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


RMS_FILES_MODE = _load_prompt("RMS_FILES_MODE.txt")
RMS_PATCH_MODE = _load_prompt("RMS_PATCH_MODE.txt")
