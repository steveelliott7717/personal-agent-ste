# backend/registry/adapters/quality.py
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from functools import lru_cache

# --- Config Loading ---


@lru_cache(maxsize=None)
def _load_config() -> Dict[str, Any]:
    """Loads and caches quality.config.json from the repo root."""
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "quality.config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# --- Process Execution Helper ---


def _run_command(command: str, cwd: Path) -> Tuple[bool, str, str]:
    """Runs a shell command and captures its output."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            timeout=300,  # 5-minute timeout
        )
        return proc.returncode == 0, proc.stdout, proc.stderr
    except FileNotFoundError:
        return False, "", f"Command not found: {command.split()[0]}"
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out after 300 seconds."
    except Exception as e:
        return False, "", f"An unexpected error occurred: {e}"


# --- Adapter Implementations ---


def quality_lint_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs linter commands based on the 'language' arg and quality.config.json.
    Args:
      language: "python" | "node" (defaults to "python")
    """
    config = _load_config()
    lang = args.get("language", "python")
    commands = (config.get(lang) or {}).get("lint", [])
    if not commands:
        return {"ok": True, "message": f"No lint commands configured for '{lang}'."}

    repo_root = Path(__file__).resolve().parents[3]
    findings = []
    for cmd in commands:
        ok, stdout, stderr = _run_command(cmd, repo_root)
        if not ok:
            findings.append({"command": cmd, "stdout": stdout, "stderr": stderr})

    if findings:
        return {
            "ok": False,
            "error": {
                "code": "LintingFailed",
                "message": f"{len(findings)} linting command(s) failed.",
                "details": findings,
            },
        }
    return {"ok": True, "message": "All lint checks passed."}


def quality_test_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs test commands based on the 'language' arg and quality.config.json.
    Args:
      language: "python" | "node" (defaults to "python")
    """
    config = _load_config()
    lang = args.get("language", "python")
    command = (config.get(lang) or {}).get("test")
    if not command:
        return {"ok": True, "message": f"No test command configured for '{lang}'."}

    repo_root = Path(__file__).resolve().parents[3]
    ok, stdout, stderr = _run_command(command, repo_root)

    if not ok:
        return {
            "ok": False,
            "error": {
                "code": "TestsFailed",
                "message": "Test suite failed.",
                "details": {"stdout": stdout, "stderr": stderr},
            },
        }
    return {"ok": True, "message": "All tests passed.", "output": stdout}


def quality_deps_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs dependency check commands.
    Args:
      language: "python" | "node" (defaults to "python")
    """
    config = _load_config()
    lang = args.get("language", "python")
    command = (config.get(lang) or {}).get("deps")
    if not command:
        return {"ok": True, "message": f"No dependency check configured for '{lang}'."}

    repo_root = Path(__file__).resolve().parents[3]
    ok, stdout, stderr = _run_command(command, repo_root)

    if not ok:
        return {
            "ok": False,
            "error": {
                "code": "VulnerabilitiesFound",
                "message": "Dependency vulnerabilities found.",
                "details": {"stdout": stdout, "stderr": stderr},
            },
        }
    return {"ok": True, "message": "No dependency vulnerabilities found."}
