# backend/agents/repo_updater_agent.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.orchestrator import Orchestrator
from backend.services.supabase_service import supabase
from backend.llm.llm_runner import run_llm_agent
from backend.registry.capability_registry import flatten_result
import time
import uuid

# --- Placeholder import ---
from backend.llm.multi_model_clients import call_reranker

# --- BEGIN: FILES→DIFF helpers ---

import difflib
import re
from pathlib import Path


def _parse_files_artifact(s: str) -> dict[str, str]:
    """
    Parse a FILES artifact:

      BEGIN_FILE path/to/file
      <entire new file content>
      END_FILE

    Returns: { "path/to/file": "<content>", ... }
    """
    files: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in (s or "").splitlines(keepends=True):
        if line.startswith("BEGIN_FILE "):
            if cur is not None:
                files[cur] = "".join(buf)
                buf.clear()
            cur = line[len("BEGIN_FILE ") :].strip()
            continue
        if line.startswith("END_FILE"):
            if cur is not None:
                files[cur] = "".join(buf)
                cur = None
                buf.clear()
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        files[cur] = "".join(buf)
    return files


def _make_unified_diff(repo_root: Path, rel_path: str, new_text: str) -> str:
    """
    Build a proper unified diff (with valid @@ hunks) from on-disk -> new text.
    """
    abs_path = repo_root / rel_path
    try:
        old_text = abs_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        old_text = ""

    a = old_text.splitlines(keepends=True)
    b = new_text.splitlines(keepends=True)
    core = "".join(
        difflib.unified_diff(
            a, b, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", n=3
        )
    )
    return core  # no extra headers; git apply is fine with ---/+++ + @@


# --- END: FILES→DIFF helpers ---


orchestrator = Orchestrator()
sb = supabase


def _strip_markdown_json(text: str) -> str:
    """Strips markdown fences from a JSON string."""
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# ---------- Supabase helpers ----------


def _start_run(
    agent_slug: str, branch: Optional[str] = None, commit: Optional[str] = None
) -> str:
    # If you created the run_start RPC, prefer that:
    # rid = sb.rpc("run_start", {"p_agent_slug": agent_slug, "p_branch": branch, "p_commit": commit}).execute().data
    # else insert directly:
    res = (
        supabase.table("agent_runs")
        .insert(
            {
                "agent_slug": agent_slug,
                "status": "running",
                "branch": branch,
                "commit_sha": commit,
            }
        )
        .execute()
    )
    return (res.data or [{}])[0]["id"]


def _finish_run(
    run_id: str, status: str, summary: Optional[str] = None, error: Optional[str] = None
) -> None:
    try:
        supabase.table("agent_runs").update(
            {
                "status": status,
                "summary": summary,
                "error": error,
                "finished_at": "now()",
            }
        ).eq("id", run_id).execute()
    except Exception:
        pass


def _log_action(
    run_id: str,
    agent_slug: str,
    verb: str,
    target: Optional[str],
    args: Any,
    result: Any,
    ok: bool,
    latency_ms: Optional[int] = None,
) -> None:
    try:
        supabase.table("agent_actions").insert(
            {
                "run_id": run_id,
                "agent_slug": agent_slug,
                "verb": verb,
                "target": target,
                "args": args,
                "result": result,
                "ok": ok,
                "latency_ms": latency_ms or 0,
            }
        ).execute()
    except Exception:
        pass


def _log_timeline_event(run_id: str, event_type: str, details: Dict[str, Any]) -> None:
    """Appends a structured event to the agent_run's timeline."""
    try:
        # Using an RPC function is the safest way to append to a JSONB array concurrently.
        # Create this function in Supabase:
        # CREATE OR REPLACE FUNCTION append_to_timeline(run_id_in uuid, event_in jsonb)
        # RETURNS void AS $$
        #   UPDATE agent_runs
        #   SET timeline = timeline || jsonb_set(event_in, '{ts}', to_jsonb(now()))
        #   WHERE id = run_id_in;
        # $$ LANGUAGE sql;
        supabase.rpc(
            "append_to_timeline",
            {"run_id_in": run_id, "event_in": {"type": event_type, **details}},
        ).execute()
    except Exception:
        pass  # Best-effort logging


def _propose_patch(run_id: str, agent_slug: str, summary: str, patch_text: str) -> str:
    # Proposals go to patch_proposals, not applied directly
    res = (
        supabase.table("patch_proposals")
        .insert(
            {
                "run_id": run_id,
                "agent_slug": agent_slug,
                "summary": summary,
                "patch": patch_text,
                "status": "pending",
            }
        )
        .execute()
    )
    return (res.data or [{}])[0]["id"]


def _revert_all_changes(run_id: str) -> Tuple[bool, str]:
    """Reverts all changes in the repo to HEAD."""
    try:
        res = orchestrator.call_verb(
            "repo.git.revert_all",
            {},
            meta={},
            correlation_id=run_id,
            idempotency_key=f"revert-all:{run_id}:{time.time()}",
        )
        if isinstance(res, dict) and res.get("ok"):
            return True, (res.get("result") or {}).get(
                "message", "Reverted successfully."
            )
        return False, (res.get("error") or {}).get("message", "Failed to revert.")
    except Exception as e:
        return False, f"Exception reverting changes: {e}"


def _commit_and_push(
    branch_name: str, commit_message: str, run_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """Commits and pushes changes to a new branch."""
    try:
        res = orchestrator.call_verb(
            "repo.commit_and_push",
            {
                "branch_name": branch_name,
                "commit_message": commit_message,
                "pr_title": commit_message,  # Use commit message for PR title
            },
            meta={},
            correlation_id=run_id,
            idempotency_key=f"commit-push:{run_id}",
        )
        if isinstance(res, dict) and res.get("ok"):
            return True, (res.get("result") or {})
        return False, {
            "error": (res.get("error") or {}).get(
                "message", "Failed to commit and push."
            )
        }
    except Exception as e:
        return False, {"error": f"Exception committing and pushing: {e}"}


def _apply_patch(patch_text: str, run_id: str) -> Tuple[bool, str]:
    """Applies a patch to the local repository via an orchestrator verb."""
    try:
        res = orchestrator.call_verb(
            "repo.patch.apply",
            {"patch_text": patch_text},
            meta={},
            correlation_id=run_id,
            idempotency_key=f"apply-patch:{run_id}:{hash(patch_text)}",
        )
        if isinstance(res, dict) and res.get("ok"):
            return True, (res.get("result") or {}).get("message", "Applied cleanly.")
        error_message = (res.get("error") or {}).get(
            "message", "Unknown error applying patch."
        )
        return False, error_message
    except Exception as e:
        return False, f"Exception applying patch: {e}"


def _run_tests(run_id: str, tests: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Runs the test suite via an orchestrator verb."""
    try:
        args = {"tests": tests} if tests else {}
        res = orchestrator.call_verb(
            "repo.test.run",
            args,
            meta={},
            correlation_id=run_id,
            idempotency_key=f"run-tests:{run_id}:{time.time()}",
        )
        if isinstance(res, dict) and res.get("ok"):
            return True, (res.get("result") or {}).get("output", "Tests passed.")
        return False, (res.get("error") or {}).get("message", "Tests failed.")
    except Exception as e:
        return False, f"Exception running tests: {e}"


def _run_linter(
    path: str, original_content: str, patch_text: str, run_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """Runs a linter on the proposed code changes."""
    try:
        res = orchestrator.call_verb(
            "repo.lint.run",
            {
                "file_path": path,
                "original_content": original_content,
                "patch_text": patch_text,
            },
            meta={},
            correlation_id=run_id,
            idempotency_key=f"lint-run:{run_id}:{hash(patch_text)}",
        )
        if isinstance(res, dict):
            return res.get("ok", False), (res.get("error") or res.get("result") or {})
        return False, {"message": "Invalid response from linter verb."}
    except Exception as e:
        return False, {"message": f"Exception running linter: {e}"}


def _run_deps_check(run_id: str) -> Tuple[bool, Dict[str, Any]]:
    """Runs a dependency vulnerability check."""
    try:
        res = orchestrator.call_verb(
            "repo.deps.check",
            {},
            meta={},
            correlation_id=run_id,
            idempotency_key=f"deps-check:{run_id}",
        )
        if isinstance(res, dict):
            return res.get("ok", False), (res.get("error") or res.get("result") or {})
        return False, {"message": "Invalid response from dependency check verb."}
    except Exception as e:
        return False, {"message": f"Exception running dependency check: {e}"}


# ---------- Repo helpers via verbs (read-only) ----------


def _read_file(path: str, run_id: str) -> Tuple[bool, str]:
    """
    Use your orchestrator's repo read verb. If you don’t have one yet,
    you can read from disk or code-host API here.
    """
    try:
        # The orchestrator expects correlation_id and idempotency_key.
        # We'll use the run_id for correlation and create a unique key for the read.
        idempotency_key = f"read-file:{run_id}:{path}"
        res = orchestrator.call_verb(
            "repo.file",
            {"path": path},
            meta={},
            correlation_id=run_id,
            idempotency_key=idempotency_key,
        )
        if isinstance(res, dict) and res.get("ok"):
            flat_res = flatten_result(res)
            return True, flat_res.get("content") or ""
    except Exception as e:
        print(f"[_read_file] Error reading file {path}: {e}")
        pass
    return False, ""


# ---------- Pipeline Stages ----------


def _run_planning_stages(
    run_id: str, agent_slug: str, task: str, change_spec: Dict[str, Any]
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[str]]]:
    """Runs the Scout, Planner, and Reranker stages."""
    # 1) Scout
    scout_input = {
        "task": task,
        "candidate_files": change_spec.get("candidate_files")
        or change_spec.get("files")
        or [],
        "hints": change_spec.get("hints") or {},
    }
    scout_result = run_llm_agent(
        agent_slug="repo_updater_scout",
        user_text=json.dumps(scout_input),
        run_id=run_id,
    )

    scout_spec = {}
    if scout_result.ok:
        try:
            clean_response = _strip_markdown_json(scout_result.response_text)
            scout_spec = json.loads(clean_response)
        except json.JSONDecodeError:
            _finish_run(run_id, "error", error="Scout returned invalid JSON")
            return None, None

    _log_action(
        run_id,
        agent_slug,
        "plan.scout",
        target=None,
        args=scout_input,
        result=scout_spec,
        ok=True,
    )
    cand = scout_spec.get("candidate_files", [])
    acceptance_tests = scout_spec.get("acceptance_tests", [])
    _log_timeline_event(
        run_id,
        "scout_completed",
        {"candidate_files": len(cand), "acceptance_tests": len(acceptance_tests)},
    )
    print(
        f"[1] Scout → {len(cand)} candidate files, {len(acceptance_tests)} acceptance tests"
    )

    if not cand:
        _finish_run(run_id, "ok", summary="no candidates; nothing to do")
        return None, None

    # 2) Planner
    planner_input = {
        "candidate_files": cand,
        "task": task,
        "hints": change_spec.get("hints") or {},
    }
    planner_result = run_llm_agent(
        agent_slug="repo_updater_planner",
        user_text=json.dumps(planner_input),
        run_id=run_id,
    )

    deep_plan = {}
    if planner_result.ok:
        try:
            clean_response = _strip_markdown_json(planner_result.response_text)
            deep_plan = json.loads(clean_response)
        except json.JSONDecodeError:
            _finish_run(run_id, "error", error="Planner returned invalid JSON")
            return None, None

    _log_action(
        run_id,
        agent_slug,
        "plan.deepen",
        target=None,
        args=planner_input,
        result=deep_plan,
        ok=True,
    )
    spans = deep_plan.get("change_spans", [])
    _log_timeline_event(run_id, "planner_completed", {"spans_generated": len(spans)})
    print(f"[2] Planner → {len(spans)} spans")

    if not spans:
        _finish_run(run_id, "ok", summary="no spans; nothing to do")
        return None, None

    # 3) Reranker
    reranker_input = {"task": task, "change_spans": spans}
    ranked_spans: List[Dict[str, Any]] = call_reranker(reranker_input)
    _log_action(
        run_id,
        agent_slug,
        "plan.rerank",
        target=None,
        args={"count": len(spans)},
        result={"top": len(ranked_spans)},
        ok=True,
    )
    _log_timeline_event(run_id, "reranker_completed", {"top_spans": len(ranked_spans)})
    print(f"[3] Reranker → {len(ranked_spans)} top spans")

    if not ranked_spans:
        _finish_run(run_id, "ok", summary="no top spans after rerank")
        return None, None

    return ranked_spans, acceptance_tests


def _run_commit_stage(
    run_id: str, agent_slug: str, task: str, successful_patches: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Applies validated patches and commits them to a new branch."""
    if not successful_patches:
        _finish_run(run_id, "ok", summary="no changes passed tests")
        print(
            f"--- Repo update pipeline END (run={run_id}) no changes passed tests ---"
        )
        return {"status": "noop", "run_id": run_id, "proposals": []}

    branch_name = f"agent-run-{run_id[:8]}"
    for patch_info in successful_patches:
        applied_ok, _ = _apply_patch(patch_info["diff_text"], run_id)
        if not applied_ok:
            _finish_run(
                run_id,
                "error",
                error=f"Failed to re-apply a validated patch to branch {branch_name}",
            )
            return {
                "status": "error",
                "run_id": run_id,
                "error": "Failed to re-apply patch",
            }

    # Final security gate: check for vulnerable dependencies before committing.
    deps_ok, deps_result = _run_deps_check(run_id)
    _log_action(
        run_id,
        agent_slug,
        "repo.deps.check",
        target=None,
        args=None,
        result=deps_result,
        ok=deps_ok,
    )
    if not deps_ok:
        error_message = f"Dependency check failed: {deps_result.get('message')}"
        _finish_run(
            run_id,
            "error",
            error=error_message,
            summary="Vulnerabilities found in dependencies.",
        )
        # Revert the applied changes since we are not proceeding.
        _revert_all_changes(run_id)
        return {
            "status": "error",
            "run_id": run_id,
            "error": error_message,
            "details": deps_result.get("details"),
        }

    commit_message = f"Apply automated changes for task: {task}"
    pushed_ok, push_result = _commit_and_push(branch_name, commit_message, run_id)

    if pushed_ok:
        branch_name = push_result.get("branch")
        pr_url = push_result.get("pr_url")
        status = "ok"
        summ = f"Committed and pushed {len(successful_patches)} changes to branch {branch_name}."
        if pr_url:
            summ += f" PR created at {pr_url}"
    else:
        status = "error"
        summ = f"Failed to push changes to branch {branch_name}: {push_result.get('error')}"

    proposal_uuids = [p["proposal_id"] for p in successful_patches]
    _finish_run(run_id, status, summary=summ)
    print(f"--- Repo update pipeline END (run={run_id}) {summ} ---")
    return {
        "status": status,
        "run_id": run_id,
        "proposals": proposal_uuids,
        "branch": push_result.get("branch") if pushed_ok else None,
        "pr_url": push_result.get("pr_url") if pushed_ok else None,
        "error": push_result.get("error") if not pushed_ok else None,
    }


def run_update_pipeline(
    task: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    apply_changes: bool = True,
    commit_branch: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrated repo update pipeline that produces perfect, git-applicable patches.

    Steps
      1) Plan (or read provided change_spans)
      2) For each span:
         - Load ORIGINAL file text from disk
         - Ask the EXECUTOR for a FILES artifact with the ENTIRE new file body
         - Validate we did not receive a snippet / diff / fenced block
         - Synthesize a unified diff (difflib) => correct @@ hunks
         - (Optional) Ask the REVIEWER; apply suggested fix if provided
      3) Concatenate diffs; (Optional) apply and commit

    Returns
      {
        "ok": bool,
        "patch": "<unified diff>",
        "events": [...],
        "applied": bool,
        "branch": str | None,
        "review": {...}  # last reviewer output
      }
    """
    events: List[Dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parents[2]

    def _ev(kind: str, **kw):
        ev = {"kind": kind, **kw}
        events.append(ev)
        return ev

    def _strip_markdown_json(text: str) -> str:
        """
        Remove ```...``` fences and leave raw payload;
        don't attempt full JSON parsing here (we have dedicated logic later).
        """
        t = (text or "").strip()
        if t.startswith("```"):
            # first fence
            i = t.find("\n")
            if i != -1:
                t = t[i + 1 :]
            # trailing fence
            if t.endswith("```"):
                t = t[:-3]
        return t.strip()

    def _parse_files_artifact(s: str) -> Dict[str, str]:
        """
        Parse a FILES artifact:

          BEGIN_FILE path/to/file
          <entire new file content>
          END_FILE

        Returns: { "path/to/file": "<content>", ... }
        """
        files: Dict[str, str] = {}
        cur: Optional[str] = None
        buf: List[str] = []
        for line in (s or "").splitlines(keepends=True):
            if line.startswith("BEGIN_FILE "):
                if cur is not None:
                    files[cur] = "".join(buf)
                    buf.clear()
                cur = line[len("BEGIN_FILE ") :].strip()
                continue
            if line.startswith("END_FILE"):
                if cur is not None:
                    files[cur] = "".join(buf)
                    cur = None
                    buf.clear()
                continue
            if cur is not None:
                buf.append(line)
        if cur is not None:
            files[cur] = "".join(buf)
        return files

    def _make_unified_diff(rel_path: str, new_text: str) -> str:
        """
        Build a proper unified diff (valid @@ hunks) from on-disk -> new text.
        """
        abs_path = repo_root / rel_path
        try:
            old_text = abs_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            old_text = ""

        a = old_text.splitlines(keepends=True)
        b = new_text.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                a, b, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", n=3
            )
        )

    def _is_destructive(diff_text: str, thr: float = 0.20) -> bool:
        removed = kept = 0
        for ln in diff_text.splitlines():
            if ln.startswith(("--- ", "+++ ", "@@ ")):
                continue
            if ln.startswith("-") and not ln.startswith("---"):
                removed += 1
            elif ln.startswith((" ", "+")):
                kept += 1
        denom = (removed + kept) or 1
        return (removed / denom) > thr

    def _merge_preserving_only_additions(original: str, model_new: str) -> str:
        """
        Keep ALL original lines and insert model additions. Never delete lines.
        """
        merged: list[str] = []
        a = original.splitlines(keepends=True)
        b = model_new.splitlines(keepends=True)
        for tag in difflib.ndiff(a, b):
            if tag.startswith("  "):  # unchanged
                merged.append(tag[2:])
            elif tag.startswith("- "):  # original-only -> KEEP
                merged.append(tag[2:])
            elif tag.startswith("+ "):  # model-only -> INSERT
                merged.append(tag[2:])
        if merged and not merged[-1].endswith("\n"):
            merged[-1] = merged[-1] + "\n"
        return "".join(merged)

    # ---------------------------
    # 1) PLAN
    # ---------------------------
    change_spans: List[Dict[str, Any]] = []
    if isinstance(task.get("change_spans"), list) and task["change_spans"]:
        change_spans = task["change_spans"]
        _ev("plan.use_provided_spans", count=len(change_spans))
    else:
        # Call the planner to produce change_spans
        planner_in = {
            "text": (
                task.get("task")
                if isinstance(task.get("task"), str)
                else json.dumps(task)
            ),
            "params": {
                "candidate_files": task.get("candidate_files") or [],
            },
        }
        plan_res = run_llm_agent(
            agent_slug="repo_updater_planner",
            input_payload=planner_in,
            run_id=run_id,
        )
        if not plan_res.ok:
            return {
                "ok": False,
                "error": f"planner failed: {plan_res.error}",
                "events": events,
            }

        # Expect minified JSON per your planner system prompt
        try:
            plan_json = json.loads(_strip_markdown_json(plan_res.response_text))
            change_spans = list(plan_json.get("change_spans") or [])
            _ev(
                "plan.generated",
                rationale=plan_json.get("rationale"),
                spans=len(change_spans),
            )
        except Exception as e:
            return {"ok": False, "error": f"bad planner JSON: {e}", "events": events}

    if not change_spans:
        return {"ok": False, "error": "no change_spans to process", "events": events}

    # ---------------------------
    # 2) EXECUTE per span (FILES -> DIFF)
    # ---------------------------
    diffs: List[str] = []
    last_review: Optional[Dict[str, Any]] = None

    for idx, span in enumerate(change_spans):
        path = (span.get("file") or "").strip()
        if not path:
            _ev("span.skip", index=idx, reason="no-file")
            continue

        # Read original file (for executor & snippet checks)
        abs_path = repo_root / path
        try:
            original = abs_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            original = ""

        # Gather context files for the executor (if provided by planner)
        context_files = span.get("context_files") or []
        ctx_map: Dict[str, str] = {}
        for c in context_files:
            cabs = repo_root / c
            try:
                ctx_map[c] = cabs.read_text(encoding="utf-8")
            except Exception:
                pass

        # Build an explicit, FILES-only prompt for the executor
        exec_instructions = [
            # Output contract
            "Return ONLY a files artifact (no diffs, no fences, no JSON).",
            "For each file:",
            "BEGIN_FILE <repo-relative-path>",
            "<entire new file content>",
            "END_FILE",
            "",
            # Preservation + anchors
            "PRESERVE ALL EXISTING CONTENT unless explicitly instructed to remove it.",
            "Make minimal edits ONLY at these anchors:",
            "- Insert the import 'from backend.logging_utils import RequestLoggingMiddleware' immediately AFTER the line 'from fastapi import FastAPI'.",
            "- Insert 'app.add_middleware(RequestLoggingMiddleware)' immediately AFTER the first occurrence of the line 'app = FastAPI()'.",
            "",
            "Do NOT restructure, reformat, or delete unrelated code.",
            "Do NOT drop existing imports, routes, routers, or config.",
            "",
            f"Target file: {path}",
            f"Intent: {span.get('intent', '')}",
        ]

        prompt_parts = [
            "\n".join(exec_instructions),
            f"\n--- ORIGINAL {path} ---\n{original}\n--- END ORIGINAL ---\n",
        ]
        if ctx_map:
            prompt_parts.append("\n--- CONTEXT FILES ---\n")
            for cpath, cbody in ctx_map.items():
                prompt_parts.append(f"### {cpath}\n{cbody}\n")

        exec_res = run_llm_agent(
            agent_slug="repo_updater_executor",
            user_text="\n".join(prompt_parts),
            run_id=run_id,
        )
        if not exec_res.ok:
            _ev("executor.error", file=path, error=exec_res.error)
            continue

        files_blob = _strip_markdown_json(exec_res.response_text)
        # Reject fenced or diff-looking payloads
        if (
            "```" in files_blob
            or "~~~" in files_blob
            or re.search(r"^(diff --git|--- a/|\+\+\+ b/|@@ )", files_blob, flags=re.M)
        ):
            _ev("executor.reject", file=path, reason="bad_format_or_diff")
            continue

        files_map = _parse_files_artifact(files_blob)
        new_body = files_map.get(path)
        if new_body is None:
            # JSON fallback: {"file_path": "...", "file_content": "..."}
            try:
                as_json = json.loads(files_blob)
                if isinstance(as_json, dict) and as_json.get("file_path") == path:
                    new_body = as_json.get("file_content")
            except Exception:
                pass

        if not isinstance(new_body, str) or not new_body.strip():
            _ev("executor.reject", file=path, reason="missing_file_body")
            continue

        # Enforce "full-file" (not a snippet) heuristic
        if len(original) > 200 and len(new_body) < 0.5 * len(original):
            # Try anchor-based synthesis instead of rejecting
            anchor_merged = None
            import_line = "from backend.logging_utils import RequestLoggingMiddleware"
            mw_line = "app.add_middleware(RequestLoggingMiddleware)"
            if path == "backend/api.py" and (
                import_line in new_body or mw_line in new_body
            ):
                lines = original.splitlines(keepends=True)
                out = []
                inserted_import = import_line in original
                for ln in lines:
                    out.append(ln)
                    if (not inserted_import) and re.match(
                        r"^from fastapi import .*FastAPI.*$", ln.strip()
                    ):
                        out.append(import_line + "\n")
                        inserted_import = True
                updated = "".join(out)
                out2 = []
                seen_app = False
                inserted_mw = mw_line in updated
                for ln in updated.splitlines(keepends=True):
                    out2.append(ln)
                    if (
                        (not inserted_mw)
                        and (not seen_app)
                        and ln.strip().startswith("app = FastAPI(")
                    ):
                        out2.append(mw_line + "\n")
                        inserted_mw = True
                        seen_app = True
                anchor_merged = "".join(out2)
            if anchor_merged:
                new_body = anchor_merged
            else:
                _ev("executor.reject", file=path, reason="likely_snippet")
                continue

        # Synthesize a correct unified diff
        diff_text = _make_unified_diff(path, new_body)
        if not diff_text.strip():
            _ev("diff.noop", file=path)
            continue

        # Auto-repair if destructive: anchor inserts for backend/api.py, else preserve+add
        if _is_destructive(diff_text, 0.20):
            import_line = "from backend.logging_utils import RequestLoggingMiddleware"
            mw_line = "app.add_middleware(RequestLoggingMiddleware)"

            if path == "backend/api.py" and (
                import_line in new_body or mw_line in new_body
            ):
                # inline anchor insertion (no imports across files)
                lines = original.splitlines(keepends=True)
                out = []
                inserted_import = import_line in original
                inserted_mw = mw_line in original
                # insert import after combined FastAPI import line
                for ln in lines:
                    out.append(ln)
                    if (not inserted_import) and re.match(
                        r"^from fastapi import .*FastAPI.*$", ln.strip()
                    ):
                        out.append(import_line + "\n")
                        inserted_import = True
                updated = "".join(out)
                if not updated.endswith("\n"):
                    updated += "\n"
                # insert middleware after first app initializer with args
                out2 = []
                seen_app = False
                for ln in updated.splitlines(keepends=True):
                    out2.append(ln)
                    if (
                        (not inserted_mw)
                        and (not seen_app)
                        and ln.strip().startswith("app = FastAPI(")
                    ):
                        out2.append(mw_line + "\n")
                        inserted_mw = True
                        seen_app = True
                merged_body = "".join(out2)
            else:
                merged_body = _merge_preserving_only_additions(original, new_body)

            repaired_diff = _make_unified_diff(path, merged_body)
            if repaired_diff.strip() and not _is_destructive(repaired_diff, 0.20):
                _ev("diff.repaired", file=path, note="converted to insert-only")
                diff_text = repaired_diff
            else:
                _ev("diff.repair_failed", file=path)

        _ev("diff.generated", file=path, bytes=len(diff_text))
        # ---------------- reviewer pass ----------------
        review_in = (
            '{"diff": ' + json.dumps(diff_text) + "}"
            if "You MUST return ONLY a minified JSON object"
            in "".join(task.get("reviewer_prompt", []))
            else diff_text
        )
        rev_res = run_llm_agent(
            agent_slug="repo_updater_reviewer",
            user_text=review_in,  # reviewer prompt expects raw diff per your settings
            run_id=run_id,
        )
        if rev_res.ok:
            try:
                last_review = json.loads(_strip_markdown_json(rev_res.response_text))
            except Exception:
                last_review = {
                    "risk": "unknown",
                    "note": rev_res.response_text,
                    "suggested_fix_diff": None,
                }
            _ev("review.done", file=path, review=last_review)
            # Accept reviewer low/medium; if suggested_fix_diff exists, prefer it
            sugg = (last_review or {}).get("suggested_fix_diff")
            if isinstance(sugg, str) and sugg.strip().startswith("--- a/"):
                diff_text = sugg
                _ev(
                    "diff.reviewer_suggested_fix_applied",
                    file=path,
                    bytes=len(diff_text),
                )
        else:
            _ev("review.error", file=path, error=rev_res.error)

        diffs.append(diff_text)

    patch_text = "\n".join(diffs).rstrip("\n") + ("\n" if diffs else "")
    if not diffs:
        return {
            "ok": True,
            "patch": "",
            "events": events,
            "applied": False,
            "review": last_review,
        }

    # ---------------------------
    # 3) Apply + Commit (optional)
    # ---------------------------
    applied = False
    branch_out = None

    if apply_changes:
        args = {
            "patch_text": patch_text,
            "allowed_paths": sorted(
                {s.get("file") for s in change_spans if s.get("file")}
            ),
        }
        res_apply = orchestrator.call_verb(
            "repo.patch.apply",
            args,
            meta={},
            correlation_id=run_id or "",
            idempotency_key=(run_id or "") + "#apply",
        )
        _ev("apply.result", ok=bool(res_apply.get("ok")), result=res_apply)
        if not res_apply.get("ok"):
            return {
                "ok": False,
                "patch": patch_text,
                "events": events,
                "applied": False,
                "apply_error": res_apply.get("error"),
            }

        applied = True

        # Commit + push if requested
        if commit_branch or commit_message:
            branch_name = commit_branch or f"feature/patch-{uuid.uuid4().hex[:8]}"
            commit_msg = commit_message or "chore: apply repo update"
            res_commit = orchestrator.call_verb(
                "repo.commit_and_push",
                {"branch_name": branch_name, "commit_message": commit_msg},
                meta={},
                correlation_id=run_id or "",
                idempotency_key=(run_id or "") + "#commit",
            )
            _ev("commit.result", ok=bool(res_commit.get("ok")), result=res_commit)
            if res_commit.get("ok"):
                branch_out = res_commit.get("branch")

    return {
        "ok": True,
        "patch": patch_text,
        "events": events,
        "applied": applied,
        "branch": branch_out,
        "review": last_review,
    }


def handle_repo_update(query: str) -> Dict[str, Any]:
    """
    Entrypoint for router_agent / HTTP. Expects a JSON ChangeSpec string:
    {
      "task": "...",
      "candidate_files": ["path/a.py", "path/b.ts"],
      "hints": {...},
      "branch": "feature/x",
      "commit": "HEAD"
    }
    """
    try:
        spec = json.loads(query)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Input must be a JSON ChangeSpec string."}
    return run_update_pipeline(spec)
