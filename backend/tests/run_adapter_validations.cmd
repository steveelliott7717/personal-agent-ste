@echo off
setlocal
set AGENTS_BASE=http://localhost:8000
python -m pytest -q tests/test_http_fetch_validations.py || exit /b 1
python -m pytest -q tests/test_browser_validations.py || exit /b 1
echo All adapter validation tests passed.
