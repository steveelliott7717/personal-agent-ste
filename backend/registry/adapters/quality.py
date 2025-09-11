# backend/registry/adapters/quality.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
from functools import lru_cache


# -------------------------------
# Config loading
# -------------------------------


@lru_cache(maxsize=None)
def _load_config() -> Dict[str, Any]:
    """
    Load and cache quality.config.json from the repo root.
    Returns {} if the file does not exist or cannot be parsed.
    """
    try:
        repo_root = Path(__file__).resolve().parents[3]
        config_path = repo_root / "quality.config.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


# -------------------------------
# Process execution helper
# -------------------------------


def _run_command(command: str, cwd: Path) -> Tuple[bool, str, str]:
    """
    Run a shell command in `cwd`, capturing stdout/stderr.
    Returns (ok, stdout, stderr). Never raises.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            timeout=300,  # 5 minutes
        )
        return proc.returncode == 0, proc.stdout, proc.stderr
    except FileNotFoundError:
        # First token is a best-effort program name
        prog = (command.split() or ["<cmd>"])[0]
        return False, "", f"Command not found: {prog}"
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out after 300 seconds."
    except Exception as e:
        return False, "", f"Unexpected error: {e}"


# -------------------------------
# Adapters
# -------------------------------


def quality_lint_adapter(args: Dict[str, Any], _meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run linter commands based on `language` and quality.config.json.

    Input:
      language: "python" | "node" (default "python")
    """
    cfg = _load_config()
    lang = str(args.get("language", "python"))
    commands: List[str] = list((cfg.get(lang) or {}).get("lint", []))  # may be []

    if not commands:
        return {"ok": True, "message": f"No lint commands configured for '{lang}'."}

    repo_root = Path(__file__).resolve().parents[3]
    failures: List[Dict[str, str]] = []

    for cmd in commands:
        ok, stdout, stderr = _run_command(cmd, repo_root)
        if not ok:
            failures.append({"command": cmd, "stdout": stdout, "stderr": stderr})

    if failures:
        return {
            "ok": False,
            "error": {
                "code": "LintingFailed",
                "message": f"{len(failures)} linting command(s) failed.",
                "details": failures,
            },
        }

    return {"ok": True, "message": "All lint checks passed."}


def quality_test_adapter(args: Dict[str, Any], _meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run test command based on `language` and quality.config.json.

    Input:
      language: "python" | "node" (default "python")
    """
    cfg = _load_config()
    lang = str(args.get("language", "python"))
    command: Optional[str] = (cfg.get(lang) or {}).get("test")

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


def quality_deps_adapter(args: Dict[str, Any], _meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run dependency/vulnerability check based on `language` and quality.config.json.

    Input:
      language: "python" | "node" (default "python")
    """
    cfg = _load_config()
    lang = str(args.get("language", "python"))
    command: Optional[str] = (cfg.get(lang) or {}).get("deps")

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
