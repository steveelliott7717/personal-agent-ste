# backend/repo/anchors.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

InsertMode = Literal["before", "after", "replace", "append_end"]


@dataclass
class Anchor:
    """
    Defines a rule for inserting or replacing content in a text file
    based on a regex pattern.
    """

    name: str
    file_glob: str
    pattern: str
    insert_mode: InsertMode = "after"
    unique: bool = True
    re_flags: int = re.MULTILINE


def apply_anchor(original_text: str, anchor: Anchor, payload: str) -> tuple[str, bool]:
    """
    Applies a single anchor rule to a text.

    Returns:
        (new_text, changed)
    """
    # Idempotency check for unique anchors
    if anchor.unique and payload in original_text:
        return original_text, False

    match = re.search(anchor.pattern, original_text, anchor.re_flags)

    if anchor.insert_mode == "append_end":
        # Always append to the end, ensuring a newline
        if not original_text.endswith("\n"):
            original_text += "\n"
        return original_text + payload + "\n", True

    if not match:
        return original_text, False

    start, end = match.span()
    prefix = original_text[:start]
    matched_text = match.group(0)
    suffix = original_text[end:]

    if anchor.insert_mode == "before":
        # Ensure payload has a trailing newline if it doesn't already
        new_payload = payload.rstrip() + "\n"
        new_text = prefix + new_payload + matched_text + suffix
    elif anchor.insert_mode == "after":
        # Ensure matched text has a trailing newline before inserting
        new_payload = payload.rstrip()
        new_text = prefix + matched_text.rstrip("\n") + "\n" + new_payload + suffix
    elif anchor.insert_mode == "replace":
        new_text = prefix + payload + suffix
    else:
        return original_text, False

    return new_text, True


def apply_many(
    original_text: str, anchors: list[Anchor], payloads: dict[str, str]
) -> tuple[str, list[str]]:
    """
    Applies multiple anchors to a text, returning the final text and a list
    of anchor names that were successfully applied.
    """
    current_text = original_text
    applied_anchors: list[str] = []

    for anchor in anchors:
        payload = payloads.get(anchor.name)
        if payload:
            new_text, changed = apply_anchor(current_text, anchor, payload)
            if changed:
                current_text = new_text
                applied_anchors.append(anchor.name)

    return current_text, applied_anchors


# --- Built-in Anchors ---

BUILTIN_ANCHORS: list[Anchor] = [
    # Python
    Anchor(
        name="python_imports",
        file_glob="*.py",
        pattern=r"^(from __future__ import annotations\n)?",
        insert_mode="after",
        unique=True,
    ),
    Anchor(
        name="fastapi_middleware",
        file_glob="*.py",
        pattern=r"^(app\s*=\s*FastAPI\(title=.*\))",
        insert_mode="after",
        unique=True,
    ),
    # JSON
    Anchor(
        name="json_config_start",
        file_glob="*.json",
        pattern=r"^{\s*\n",
        insert_mode="after",
        unique=True,
    ),
    Anchor(
        name="json_config_end",
        file_glob="*.json",
        pattern=r"\n}\s*$",
        insert_mode="before",
        unique=True,
    ),
    # TypeScript
    Anchor(
        name="ts_imports",
        file_glob="*.ts",
        pattern=r"^(import.*from.*;\n)+",
        insert_mode="after",
        unique=True,
    ),
]


def find_anchor(name: str) -> Optional[Anchor]:
    """Finds a built-in anchor by name."""
    return next((a for a in BUILTIN_ANCHORS if a.name == name), None)
