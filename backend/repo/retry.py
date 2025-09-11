# backend/repo/retry.py
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Tuple
from typing import Optional

from backend.repo.prompts import RMS_FILES_MODE, RMS_PATCH_MODE
from backend.registry.anchors import BUILTIN_ANCHORS, apply_many

logger = logging.getLogger(__name__)

# Simple regex to check for common diff headers
_DIFF_START_RE = re.compile(r"^(--- a/|\+\+\+ b/|diff --git)", re.MULTILINE)


@dataclass
class RetryPolicy:
    """Defines the policy for retrying a failing LLM call."""

    max_attempts: int = 3
    # Delays in seconds for each retry attempt (len should be max_attempts - 1)
    backoff_seconds: List[int] = field(default_factory=lambda: [0, 2, 5])


def _has_markdown_fences(text: str) -> bool:
    """Checks for ```...``` code fences."""
    return text.strip().startswith("```") and text.strip().endswith("```")


def _strip_markdown_fences(text: str) -> str:
    """Strips markdown fences and optional language hint from a string."""
    s = text.strip()
    if s.startswith("```"):
        first_line_end = s.find("\n")
        if first_line_end != -1:
            s = s[first_line_end + 1 :]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _is_malformed_diff(text: str) -> bool:
    """A simple check to see if a text that should be a diff is malformed."""
    # A valid diff should have headers and at least one '@@' hunk marker.
    has_headers = _DIFF_START_RE.search(text)
    has_hunks = "@@ " in text
    return bool(has_headers and not has_hunks)


def _is_snippet(original_content: Optional[str], new_content: str) -> bool:
    """Heuristic to detect if the model returned a snippet instead of a full file."""
    if not original_content or not new_content:
        return False
    # A snippet is likely if the original was large and the new content is much smaller.
    return len(original_content) > 200 and len(new_content) < 0.5 * len(
        original_content
    )


def _log_retry(reason: str, attempt: int, mode: str, **kwargs):
    """Placeholder for structured telemetry logging."""
    logger.warning(
        f"[retry] reason='{reason}' attempt={attempt} mode='{mode}' details={kwargs}"
    )


def retry_artifact_generation(
    llm_call: Callable[[str, str], str],
    initial_mode: str,
    initial_user_prompt: str,
    original_content: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Handles the retry loop for generating a FILES or PATCH artifact.
    Switches from PATCH to FILES mode on certain failures.
    Returns (final_output, final_mode, resolution_method).
    """
    policy = RetryPolicy()
    current_mode = initial_mode
    last_output: str = ""
    attempt = 0

    while attempt < policy.max_attempts:
        if attempt > 0 and policy.backoff_seconds:
            time.sleep(
                policy.backoff_seconds[
                    min(attempt - 1, len(policy.backoff_seconds) - 1)
                ]
            )

        system_prompt = RMS_FILES_MODE if current_mode == "files" else RMS_PATCH_MODE
        last_output = llm_call(system_prompt, initial_user_prompt)

        # --- Validation and Mode Switching ---

        if _has_markdown_fences(last_output):
            _log_retry("markdown_fences", attempt + 1, current_mode)
            last_output = _strip_markdown_fences(last_output)

        if current_mode == "patch" and _is_malformed_diff(last_output):
            _log_retry(
                "malformed_diff", attempt + 1, current_mode, switching_to="files"
            )
            current_mode = "files"
            attempt += 1
            continue  # Retry with the new mode

        # If we are in 'files' mode and the output is still a malformed diff, it's a failure.
        if current_mode == "files" and _is_malformed_diff(last_output):
            _log_retry("malformed_diff_in_files_mode", attempt + 1, current_mode)
            attempt += 1
            continue  # Try again in files mode

        # Snippet detection and anchor-based repair for FILES mode
        if current_mode == "files" and _is_snippet(original_content, last_output):
            _log_retry(
                "snippet_detected",
                attempt + 1,
                current_mode,
                switching_to="anchor_repair",
            )
            if original_content:
                # The snippet is the payload for the anchors
                payloads = {anchor.name: last_output for anchor in BUILTIN_ANCHORS}
                repaired_content, applied_anchors = apply_many(
                    original_content, BUILTIN_ANCHORS, payloads
                )
                if applied_anchors:
                    _log_retry(
                        "anchor_repair_successful",
                        attempt + 1,
                        current_mode,
                        applied=applied_anchors,
                    )
                    # Return the repaired content, not the snippet
                    return repaired_content, "files", "anchor_repair"

            # If repair fails, log it and continue the retry loop for a new LLM response
            _log_retry("anchor_repair_failed", attempt + 1, current_mode)
            attempt += 1
            continue

        # If we've passed all checks for the current mode, the output is considered valid.
        return last_output, current_mode, "llm_direct"

        attempt += 1

    # If all attempts fail, return the last output and mode.
    return last_output, current_mode, "failed"
