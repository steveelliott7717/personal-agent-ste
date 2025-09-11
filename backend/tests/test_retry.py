# tests/test_retry.py
import unittest
from unittest.mock import MagicMock, patch

from backend.repo.retry import retry_artifact_generation


class TestRetryLogic(unittest.TestCase):
    def test_no_retry_on_success(self):
        """Should return the first response if it's valid."""
        llm_call = MagicMock(return_value="BEGIN_FILE a.py\nprint('ok')\nEND_FILE")
        output, mode, _resolution = retry_artifact_generation(
            llm_call, "files", "test task"
        )
        self.assertEqual(output, "BEGIN_FILE a.py\nprint('ok')\nEND_FILE")
        self.assertEqual(mode, "files")
        llm_call.assert_called_once()

    def test_strips_markdown_fences(self):
        """Should strip markdown and succeed on the first attempt."""
        fenced_response = "```\nBEGIN_FILE a.py\nprint('ok')\nEND_FILE\n```"
        llm_call = MagicMock(return_value=fenced_response)
        output, mode, _resolution = retry_artifact_generation(
            llm_call, "files", "test task"
        )
        self.assertEqual(output, "BEGIN_FILE a.py\nprint('ok')\nEND_FILE")
        self.assertEqual(mode, "files")
        llm_call.assert_called_once()

    def test_malformed_patch_retries_as_files(self):
        """Should detect a bad patch, switch to 'files' mode, and succeed."""
        bad_patch = "--- a/a.py\n+++ b/a.py"  # Missing hunk
        good_files = "BEGIN_FILE a.py\nprint('ok')\nEND_FILE"
        llm_call = MagicMock(side_effect=[bad_patch, good_files])

        output, mode, _resolution = retry_artifact_generation(
            llm_call, "patch", "test task"
        )

        self.assertEqual(output, good_files)
        self.assertEqual(mode, "files")
        self.assertEqual(llm_call.call_count, 2)

    def test_exhausts_retries(self):
        """Should return the last bad output after all retries are exhausted (3 attempts)."""
        # Always return a malformed patch
        bad_patch = "--- a/a.py\n+++ b/a.py"
        llm_call = MagicMock(return_value=bad_patch)

        # The retry_artifact_generation function uses a default RetryPolicy.
        # We can't easily inject a different one without changing the function signature.
        # So, we'll test against the default policy's `max_attempts=3`.
        with patch("backend.repo.retry.time.sleep"):  # Avoid sleeping during tests
            output, mode, _resolution = retry_artifact_generation(
                llm_call, "patch", "test task"
            )

        self.assertEqual(output, bad_patch)
        self.assertEqual(mode, "files")  # It will have switched to files mode
        self.assertEqual(llm_call.call_count, 3)  # Default policy has 3 attempts
