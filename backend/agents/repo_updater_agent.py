# backend/agents/repo_updater_agent.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.orchestrator import Orchestrator
from backend.services.supabase_service import supabase
from backend.llm.llm_runner import run_llm_agent
import time

# --- Placeholder import ---
from backend.llm.multi_model_clients import call_reranker


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
            return True, (res.get("result") or {}).get("content") or ""
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


def run_update_pipeline(change_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates a multi-LLM repo update:
    1) Scout (Gemini Flash) -> candidate_files
    2) Planner (Gemini Pro)  -> change_spans (file + anchors + intent)
    3) Reranker               -> top spans
    4) Executor (GPT-5 Pro)   -> unified diff (one file per call)
    5) Reviewer (Claude)      -> risk JSON
    6) Propose diff to Supabase (no direct apply)
    """
    agent_slug = "repo_updater"
    task = change_spec.get("task") or "repo update"
    branch = change_spec.get("branch")
    commit = change_spec.get("commit")

    run_id = _start_run(agent_slug, branch=branch, commit=commit)
    print(f"--- Repo update pipeline START (run={run_id}) task={task!r} ---")

    try:
        # Stages 1-3: Scout, Plan, Rerank
        ranked_spans, acceptance_tests = _run_planning_stages(
            run_id, agent_slug, task, change_spec
        )
        if ranked_spans is None:
            return {
                "status": "noop",
                "run_id": run_id,
                "message": "Planning stages did not produce any work.",
            }

        # --- Test & Collect Loop ---
        successful_patches: List[Dict[str, str]] = []
        for span in ranked_spans:
            path = span.get("file")
            if not path:
                continue

            ok_read, file_text = _read_file(path, run_id=run_id)
            if not ok_read or not file_text:
                _log_action(
                    run_id,
                    agent_slug,
                    "repo.file.read",
                    target=path,
                    args=None,
                    result={"ok": False},
                    ok=False,
                )
                continue

            # GPT-5 Pro: generate unified diff
            executor_input = {
                "file_path": path,
                "file_content": file_text,
                "intent": span.get("intent"),
            }
            executor_result = run_llm_agent(
                agent_slug="repo_updater_executor",
                user_text=json.dumps(executor_input),
                run_id=run_id,
            )
            diff_text = (
                _strip_markdown_json(executor_result.response_text)
                if executor_result.ok
                else ""
            )
            _log_action(
                run_id,
                agent_slug,
                "repo.diff.generate",
                target=path,
                args={"span": span},
                result={"bytes": len(diff_text)},
                ok=bool(diff_text),
            )

            if not diff_text.strip().startswith("--- a/"):
                # strict guard: only accept proper diffs
                _log_action(
                    run_id,
                    agent_slug,
                    "repo.diff.reject",
                    target=path,
                    args=None,
                    result={"reason": "invalid diff header"},
                    ok=False,
                )
                continue

            # NEW: Lint the generated code before review
            lint_ok, lint_result = _run_linter(path, file_text, diff_text, run_id)
            _log_action(
                run_id,
                agent_slug,
                "repo.lint.run",
                target=path,
                args=None,
                result=lint_result,
                ok=lint_ok,
            )

            if not lint_ok:
                print(f"[!] Linting failed for patch on {path}. Attempting to fix.")
                fixer_input = {
                    "file_path": path,
                    "file_content": file_text,
                    "failed_diff": diff_text,
                    "linting_errors": lint_result.get("details"),
                    "intent": span.get("intent"),
                }
                fixer_result = run_llm_agent(
                    agent_slug="repo_updater_fixer",
                    user_text=json.dumps(fixer_input),
                    run_id=run_id,
                )

                if fixer_result.ok and (
                    fixed_diff := _strip_markdown_json(fixer_result.response_text)
                ).strip().startswith("--- a/"):
                    print(
                        "[*] Fixer agent proposed a new diff based on linting errors."
                    )
                    diff_text = fixed_diff  # Update the diff with the fixed version
                    _log_timeline_event(run_id, "lint_fix_proposed", {"path": path})
                    _log_action(
                        run_id,
                        agent_slug,
                        "repo.diff.fix",
                        target=path,
                        args=fixer_input,
                        result={"bytes": len(diff_text)},
                        ok=True,
                    )

                    # Re-lint the fixed diff to ensure it's clean now
                    lint_ok, lint_result = _run_linter(
                        path, file_text, diff_text, run_id
                    )
                    _log_action(
                        run_id,
                        agent_slug,
                        "repo.lint.run",
                        target=path,
                        args={"attempt": "fix"},
                        result=lint_result,
                        ok=lint_ok,
                    )
                    if not lint_ok:
                        print("[!] Linting failed again after fix. Discarding change.")
                        continue
                else:
                    print(
                        "[!] Fixer agent failed to fix linting errors. Discarding change."
                    )
                    _log_timeline_event(run_id, "lint_fix_failed", {"path": path})
                    _log_action(
                        run_id,
                        agent_slug,
                        "repo.diff.fix.fail",
                        target=path,
                        args=fixer_input,
                        result={"error": fixer_result.error or "invalid diff header"},
                        ok=False,
                    )
                    continue

            # Claude: review
            reviewer_result = run_llm_agent(
                agent_slug="repo_updater_reviewer",
                user_text=diff_text,
                # The model is configured in agent_settings, falling back to llm_runner default.
                run_id=run_id,
            )
            review = {}
            if reviewer_result.ok:
                try:
                    clean_response = _strip_markdown_json(reviewer_result.response_text)
                    review = json.loads(clean_response)
                except json.JSONDecodeError:
                    review = {"risk": "high", "note": "Reviewer returned invalid JSON."}
            _log_action(
                run_id,
                agent_slug,
                "repo.diff.review",
                target=path,
                args=None,
                result=review,
                ok=True,
            )
            _log_timeline_event(
                run_id,
                "review_completed",
                {"path": path, "risk": review.get("risk", "unknown")},
            )

            # Test & Fix Loop
            if (review.get("risk") or "high").lower() == "low":
                current_diff = diff_text
                fix_attempts = 0
                max_fix_attempts = 2  # Allow 2 attempts to fix a failing patch

                while fix_attempts <= max_fix_attempts:
                    applied_ok, apply_msg = _apply_patch(current_diff, run_id)
                    _log_action(
                        run_id,
                        agent_slug,
                        "repo.patch.apply",
                        target=path,
                        args={"attempt": fix_attempts + 1},
                        result={"message": apply_msg},
                        ok=applied_ok,
                    )

                    if not applied_ok:
                        print(
                            f"[!] Patch failed to apply for {path}. Discarding change."
                        )
                        break  # If it can't even apply, no point in trying to fix.

                    tests_ok, test_output = _run_tests(run_id, tests=acceptance_tests)
                    _log_action(
                        run_id,
                        agent_slug,
                        "repo.test.run",
                        target=None,
                        args={"attempt": fix_attempts + 1},
                        result={"output": test_output},
                        ok=tests_ok,
                    )

                    # IMPORTANT: Always revert to keep the working dir clean for the next span/fix attempt
                    _revert_all_changes(run_id)

                    if tests_ok:
                        print(f"[✓] Tests passed for patch on {path}.")
                        _log_timeline_event(
                            run_id,
                            "tests_passed",
                            {"path": path, "attempt": fix_attempts + 1},
                        )
                        proposal_id = _propose_patch(
                            run_id,
                            agent_slug,
                            span.get("intent", f"Update {path}"),
                            current_diff,
                        )
                        successful_patches.append(
                            {"proposal_id": proposal_id, "diff_text": current_diff}
                        )
                        break  # Exit the fix loop on success
                    else:
                        fix_attempts += 1
                        if fix_attempts > max_fix_attempts:
                            print(
                                f"[!] Tests failed after {max_fix_attempts} fix attempts for patch on {path}. Discarding change."
                            )
                            _log_action(
                                run_id,
                                agent_slug,
                                "repo.test.fail.final",
                                target=None,
                                args={"output": test_output},
                                result=None,
                                ok=False,
                            )
                            _log_timeline_event(
                                run_id, "tests_failed_final", {"path": path}
                            )
                            break

                        print(
                            f"[!] Tests failed for patch on {path}. Attempting to fix (attempt {fix_attempts}/{max_fix_attempts})."
                        )
                        _log_action(
                            run_id,
                            agent_slug,
                            "repo.test.fail",
                            target=None,
                            args={"output": test_output},
                            result=None,
                            ok=False,
                        )
                        _log_timeline_event(
                            run_id,
                            "tests_failed_retrying",
                            {"path": path, "attempt": fix_attempts},
                        )

                        # Call the fixer agent
                        fixer_input = {
                            "file_path": path,
                            "file_content": file_text,
                            "failed_diff": current_diff,
                            "test_failure_output": test_output,
                            "intent": span.get("intent"),
                        }
                        fixer_result = run_llm_agent(
                            agent_slug="repo_updater_fixer",
                            user_text=json.dumps(fixer_input),
                            run_id=run_id,
                        )

                        if fixer_result.ok and (
                            fixed_diff := _strip_markdown_json(
                                fixer_result.response_text
                            )
                        ).strip().startswith("--- a/"):
                            print("[*] Fixer agent proposed a new diff.")
                            current_diff = fixed_diff  # Update the diff to be tested in the next loop iteration
                            _log_timeline_event(
                                run_id, "test_fix_proposed", {"path": path}
                            )
                            _log_action(
                                run_id,
                                agent_slug,
                                "repo.diff.fix",
                                target=path,
                                args=fixer_input,
                                result={"bytes": len(current_diff)},
                                ok=True,
                            )
                        else:
                            print(
                                "[!] Fixer agent returned invalid diff or failed. Discarding change."
                            )
                            _log_timeline_event(
                                run_id, "test_fix_failed", {"path": path}
                            )
                            _log_action(
                                run_id,
                                agent_slug,
                                "repo.diff.fix.fail",
                                target=path,
                                args=fixer_input,
                                result={
                                    "error": fixer_result.error or "invalid diff header"
                                },
                                ok=False,
                            )
                            break
            else:
                _log_action(
                    run_id,
                    agent_slug,
                    "repo.diff.skip",
                    target=path,
                    args=None,
                    result={"reason": "high_risk", "review": review},
                    ok=False,
                )
                _log_timeline_event(run_id, "change_skipped_high_risk", {"path": path})

        # Final Stage: Commit and Push
        return _run_commit_stage(run_id, agent_slug, task, successful_patches)

    except Exception as e:
        _finish_run(run_id, "error", error=f"{e.__class__.__name__}: {e}")
        return {
            "status": "error",
            "run_id": run_id,
            "error": f"{e.__class__.__name__}: {e}",
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
