# tests/test_quality_adapter.py
import unittest
from unittest.mock import patch
import json
from pathlib import Path

from backend.registry.adapters.quality import (
    quality_lint_adapter,
    quality_test_adapter,
    quality_deps_adapter,
    _load_config,
)


class TestQualityAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a dummy config for tests
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "quality.config.json"
        cls.dummy_config = {
            "python": {
                "lint": ["echo 'lint ok'"],
                "test": "echo 'test ok'",
                "deps": "echo 'deps ok'",
            },
            "python_fail": {
                "lint": ["exit 1"],
                "test": "exit 1",
                "deps": "exit 1",
            },
        }
        with open(cls.config_path, "w") as f:
            json.dump(cls.dummy_config, f)
        # Force reload of config
        _load_config.cache_clear()

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy config
        if cls.config_path.exists():
            cls.config_path.unlink()

    @patch("backend.registry.adapters.quality._run_command")
    def test_lint_success(self, mock_run_command):
        mock_run_command.return_value = (True, "Linting passed", "")
        result = quality_lint_adapter({"language": "python"}, {})
        self.assertTrue(result["ok"])
        self.assertEqual(result["message"], "All lint checks passed.")

    @patch("backend.registry.adapters.quality._run_command")
    def test_lint_failure(self, mock_run_command):
        mock_run_command.return_value = (False, "", "Lint error found")
        result = quality_lint_adapter({"language": "python"}, {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "LintingFailed")
        self.assertIn("linting command(s) failed", result["error"]["message"])

    @patch("backend.registry.adapters.quality._run_command")
    def test_test_success(self, mock_run_command):
        mock_run_command.return_value = (True, "All tests passed.", "")
        result = quality_test_adapter({"language": "python"}, {})
        self.assertTrue(result["ok"])
        self.assertEqual(result["message"], "All tests passed.")

    @patch("backend.registry.adapters.quality._run_command")
    def test_test_failure(self, mock_run_command):
        mock_run_command.return_value = (False, "", "1 test failed")
        result = quality_test_adapter({"language": "python"}, {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "TestsFailed")

    @patch("backend.registry.adapters.quality._run_command")
    def test_deps_failure(self, mock_run_command):
        mock_run_command.return_value = (False, "Vulnerability found", "")
        result = quality_deps_adapter({"language": "python"}, {})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VulnerabilitiesFound")
