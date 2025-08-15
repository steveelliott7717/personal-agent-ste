# RMS GPT — Operator Runbook (Improved Output Discipline)

## Endpoints (unchanged)
- `POST /app/api/repo/plan` — returns a preview artifact. Add `?format=files|patch` to get raw cargo.
- `POST /app/api/repo/files` — returns **BEGIN_FILE/END_FILE** blocks (files mode only).

## Pinned Prompts
Use these as system prompts:
- `prompts/RMS_FILES_MODE.txt` for files mode
- `prompts/RMS_PATCH_MODE.txt` for patch mode

When building the user message, prepend:
```
PATH_PREFIX=<your path prefix>
MODE=files|patch
```

## Files Mode — Acceptance
1) Response contains only `BEGIN_FILE/END_FILE` blocks.
2) ASCII-only, LF-only, exactly one trailing LF per file.
3) Paths all start with `PATH_PREFIX`.
4) If `backend/main.py` is present, ensure:
   - ONE `from backend.logging_utils import setup_logging, RequestLoggingMiddleware`
   - ONE `setup_logging()` *immediately after* `app = FastAPI(title="Personal Agent API")`
   - ONE `app.add_middleware(RequestLoggingMiddleware)`

Validate:
```bash
python tools/test_files_validate.py files.out
```

## Patch Mode — Acceptance
1) First non-empty line: `diff --git a/... b/...`
2) Headers in correct order; new files use `/dev/null`.
3) Only valid hunk line prefixes.
4) All b/<path> targets start with PATH_PREFIX.
5) ASCII + LF; exactly one trailing LF.

Validate quickly:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/test_patch_apply.ps1 patch.out
```

## Common Pitfalls & Fixes
- **Markdown fences in output**: remove via `backend/utils/patch_sanitizer.sanitize_patch`.
- **CRLF endings**: always normalize to LF before validation.
- **Duplicate headers**: reject patch; ask model for a clean re-emit with the pinned prompt.

## Minimal Workflow
```bash
# Files mode
curl -sS -X POST "$HOST/app/api/repo/files" -H "Content-Type: application/json" --data-binary @payload.json > files.out
python tools/test_files_validate.py files.out

# Patch mode
curl -sS -X POST "$HOST/app/api/repo/plan?format=patch" -H "Content-Type: application/json" --data-binary @payload.json > patch.out
powershell -NoProfile -ExecutionPolicy Bypass -File tools/test_patch_apply.ps1 patch.out
```
