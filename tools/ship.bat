@echo off
setlocal ENABLEEXTENSIONS

rem --- usage ---
if "%~1"=="" goto :usage
set MSG=%*

echo ===== SHIP (cmd.exe) =====
echo Commit: %MSG%
echo ==========================

rem --- must run from repo root ---
if not exist .git goto :noroot

rem --- stash unstaged work to avoid pre-commit conflicts ---
set NEEDSTASH=
git diff --quiet --ignore-submodules HEAD || set NEEDSTASH=1
if defined NEEDSTASH git stash push -u -m ship-auto-stash >nul 2>&1

rem --- stage everything you want to ship ---
echo [+] git add -A
git add -A || goto :fail

rem --- run pre-commit (fallback to python -m); re-stage after hooks ---
echo [+] pre-commit run (staged only)
pre-commit run
if errorlevel 1 (
  echo [i] pre-commit returned non-zero (fixed files or not on PATH). Re-staging...
  git add -A
  python -m pre-commit run
  git add -A
)

rem --- commit/push/deploy ---
echo [+] git commit
git commit -m "%MSG%" || goto :restore

echo [+] git push
git push || goto :fail

echo [+] fly deploy
fly deploy || goto :fail

echo [OK] Shipped and deployed successfully.
goto :restore

:fail
echo [ERROR] Command failed. Aborting.

:restore
if defined NEEDSTASH git stash pop >nul 2>&1
endlocal
exit /b 0

:usage
echo Usage: tools\ship.bat "commit message"
endlocal
exit /b 2

:noroot
echo [ERROR] Run from the repo root (where .git exists).
endlocal
exit /b 1
