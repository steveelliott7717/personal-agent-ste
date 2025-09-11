# backend/repo/retry.py
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from backend.repo.prompts import RMS_FILES_MODE, RMS_PATCH_MODE

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


def _log_retry(reason: str, attempt: int, mode: str, **kwargs):
    """Placeholder for structured telemetry logging."""
    logger.warning(
        f"[retry] reason='{reason}' attempt={attempt} mode='{mode}' details={kwargs}"
    )


def retry_artifact_generation(
    llm_call: Callable[[str, str], str],
    initial_mode: str,
    initial_user_prompt: str,
) -> Tuple[str, str]:
    """
    Handles the retry loop for generating a FILES or PATCH artifact.
    Switches from PATCH to FILES mode on certain failures.
    """
    policy = RetryPolicy()
    current_mode = initial_mode
    last_output = ""

    for attempt in range(policy.max_attempts):
        if attempt > 0 and policy.backoff_seconds:
            time.sleep(
                policy.backoff_seconds[
                    min(attempt - 1, len(policy.backoff_seconds) - 1)
                ]
            )

        system_prompt = RMS_FILES_MODE if current_mode == "files" else RMS_PATCH_MODE
        last_output = llm_call(system_prompt, initial_user_prompt)

        if _has_markdown_fences(last_output):
            _log_retry("markdown_fences", attempt, current_mode)
            last_output = _strip_markdown_fences(last_output)

        if current_mode == "patch" and _is_malformed_diff(last_output):
            _log_retry("malformed_diff", attempt, current_mode, switching_to="files")
            current_mode = "files"
            continue  # Retry with the new mode

        # If we've passed all checks for the current mode, it's valid.
        return (str(last_output), str(current_mode))

    # If all attempts fail, return the last output and mode.
    return (str(last_output), str(current_mode))
