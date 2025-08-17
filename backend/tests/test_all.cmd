@echo off
setlocal ENABLEDELAYEDEXPANSION

REM change to this script's folder
cd /d %~dp0

set BASE=http://localhost:8000/app/api/agents/verb
echo Running DB read test suite against %BASE%
echo.

set FAIL=0

call :run python -u test_dbread_operators.py
call :run python -u test_dbread_groups.py
call :run python -u test_dbread_negations.py
call :run python -u test_dbread_edgecases.py
call :run python -u test_dbread_expand.py
call :run python -u test_dbread_count.py
call :run python -u test_dbread_suite.py

REM optional: include the larger smokes if you saved them too
if exist smoke_dbread_groups_full.py call :run python -u smoke_dbread_groups_full.py
if exist smoke_dbread_triples.py     call :run python -u smoke_dbread_triples.py
if exist smoke_dbread_edges.py       call :run python -u smoke_dbread_edges.py

echo.
if %FAIL% NEQ 0 (
  echo === ONE OR MORE TEST FILES REPORTED FAILURES ===
  exit /b 1
) else (
  echo === ALL TEST FILES COMPLETED ===
  exit /b 0
)

:run
echo ------------------------------------------------------------
echo %*
echo ------------------------------------------------------------
%*
if errorlevel 1 (
  set /a FAIL=FAIL+1
  echo (test file returned non-zero exit)
)
echo.
exit /b 0
