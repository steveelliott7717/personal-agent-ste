@echo off
setlocal ENABLEDELAYEDEXPANSION

rem ================== Setup ==================
chcp 65001 >nul
if "%HOST%"=="" set HOST=http://localhost:8000
set OUTDIR=%TEMP%\pa_tests
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

rem NOTE: removed --compressed because your curl build doesn't support it
set CURL=curl -sS --connect-timeout 3 --max-time 10 -H "Accept: application/json"
set VERBURL=%HOST%/app/api/agents/verb
set PY=python
set SUMMARY=%PY% -c "import sys,json; d=json.load(sys.stdin); r=d.get('result') or {}; print(json.dumps({'ok':d.get('ok'),'rows':r.get('rows'),'meta':r.get('meta'),'error':d.get('error')}, indent=2, ensure_ascii=False))"
set PYTHONIOENCODING=utf-8

echo(
echo =============================================
echo Personal Agent API - Batch Test Suite
echo HOST = %HOST%
echo Temp dir = %OUTDIR%
echo =============================================
echo(

rem ================== Health ==================
echo ---- [HEALTH] -------------------------------------------
%CURL% "%HOST%/app/api/health"
echo(

rem ===== Helper to POST a JSON file and show concise summary with error guards =====
goto :AFTER_FUNCS
:DO_POST
  echo ---- [POST] %LABEL% ------------------------------------
  echo File: %OUTDIR%\%FILE%

  set "RESP=%OUTDIR%\resp_%RANDOM%.json"
  set "ERRF=%RESP%.err"

  rem Run curl; stdout -> RESP, stderr -> ERRF
  %CURL% -X POST "%VERBURL%" -H "Content-Type: application/json" --data "@%OUTDIR%\%FILE%" 1>"%RESP%" 2>"%ERRF%"
  if errorlevel 1 (
    echo CURL ERROR:
    type "%ERRF%"
    del /q "%ERRF%" >nul 2>&1
    del /q "%RESP%" >nul 2>&1
    echo(
    goto :eof
  )

  rem Ensure we got some body
  for %%A in ("%RESP%") do set SZ=%%~zA
  if "%SZ%"=="0" (
    echo Empty response body.
    del /q "%RESP%" "%ERRF%" >nul 2>&1
    echo(
    goto :eof
  )

  rem Pretty, compact summary
  %SUMMARY% < "%RESP%"

  del /q "%RESP%" "%ERRF%" >nul 2>&1
  echo(
  goto :eof
:AFTER_FUNCS

rem ================== 0) COUNT ==================
> "%OUTDIR%\t01_count.json" echo {"verb":"db.read","args":{"table":"events","limit":0,"aggregate":{"count":"*"}}}
set LABEL=COUNT_ALL
set FILE=t01_count.json
call :DO_POST

rem ================== 1) PATTERN / REGEX (unified) ==================
> "%OUTDIR%\t10_contains.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"contains","value":"plan"}},"limit":1}}
set LABEL=CONTAINS
set FILE=t10_contains.json
call :DO_POST

> "%OUTDIR%\t11_icontains.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"icontains","value":"plan"}},"limit":1}}
set LABEL=ICONTAINS
set FILE=t11_icontains.json
call :DO_POST

> "%OUTDIR%\t12_starts.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"starts_with","value":"plan."}},"limit":1}}
set LABEL=STARTS_WITH
set FILE=t12_starts.json
call :DO_POST

rem Use a likely hit value; change back to "ERROR" if you want a miss case
> "%OUTDIR%\t13_ends.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"iends_with","value":"start"}},"limit":1}}
set LABEL=IENDS_WITH
set FILE=t13_ends.json
call :DO_POST

rem IMPORTANT: escape % as %% in batch files
> "%OUTDIR%\t14_ilike.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"ilike","value":"plan.%%"}},"limit":1}}
set LABEL=ILIKE_DIRECT
set FILE=t14_ilike.json
call :DO_POST

> "%OUTDIR%\t15_imatch.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"imatch","value":"^plan\\..*"}},"limit":1}}
set LABEL=IMATCH_REGEX
set FILE=t15_imatch.json
call :DO_POST

rem ================== 2) NOT wrapper (patterns) ==================
> "%OUTDIR%\t20_not_icontains.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"op":"not","conditions":[{"field":"topic","op":"icontains","value":"plan"}]},"limit":1}}
set LABEL=NOT_ICONTAINS
set FILE=t20_not_icontains.json
call :DO_POST

> "%OUTDIR%\t21_not_imatch.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"op":"not","conditions":[{"field":"topic","op":"imatch","value":"^plan\\..*"}]},"limit":1}}
set LABEL=NOT_IMATCH
set FILE=t21_not_imatch.json
call :DO_POST

rem ================== 3) *_any sugars ==================
> "%OUTDIR%\t30_starts_any.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"starts_with_any","value":["plan.","verb."]}},"limit":1}}
set LABEL=STARTS_WITH_ANY
set FILE=t30_starts_any.json
call :DO_POST

> "%OUTDIR%\t31_iends_any.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"topic":{"op":"iends_with_any","value":["start","result"]}},"limit":1}}
set LABEL=IENDS_WITH_ANY
set FILE=t31_iends_any.json
call :DO_POST

> "%OUTDIR%\t32_icontains_any.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"payload.text":{"op":"icontains_any","value":["fail","panic"]}},"limit":1}}
set LABEL=ICONTAINS_ANY_JSONPATH
set FILE=t32_icontains_any.json
call :DO_POST

rem ================== 4) Between / Not Between ==================
> "%OUTDIR%\t40_between.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"latency_ms":{"op":"between","value":[10,50]}},"limit":1}}
set LABEL=BETWEEN_INCLUSIVE
set FILE=t40_between.json
call :DO_POST

> "%OUTDIR%\t41_between_excl.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"latency_ms":{"op":"between_exclusive","value":[10,50]}},"limit":1}}
set LABEL=BETWEEN_EXCLUSIVE
set FILE=t41_between_excl.json
call :DO_POST

> "%OUTDIR%\t42_not_between.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","latency_ms","created_at"],"where":{"latency_ms":{"op":"not_between","value":[10,50]}},"limit":1}}
set LABEL=NOT_BETWEEN
set FILE=t42_not_between.json
call :DO_POST

rem ================== 5) JSON key presence / null helpers ==================
> "%OUTDIR%\t50_exists.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"payload.latency_ms":{"op":"exists"}},"limit":1}}
set LABEL=JSON_EXISTS
set FILE=t50_exists.json
call :DO_POST

> "%OUTDIR%\t51_isnull.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"topic":{"op":"is_null"}},"limit":1}}
set LABEL=IS_NULL
set FILE=t51_isnull.json
call :DO_POST

> "%OUTDIR%\t52_notnull.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"topic":{"op":"not_null"}},"limit":1}}
set LABEL=NOT_NULL
set FILE=t52_notnull.json
call :DO_POST

rem ================== 6) IN / NOT IN (top-level only) ==================
> "%OUTDIR%\t60_in.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"source_agent":{"op":"in","value":["manual","orchestrator.meta"]}},"limit":1}}
set LABEL=IN_TOPLEVEL
set FILE=t60_in.json
call :DO_POST

> "%OUTDIR%\t61_not_in.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"source_agent":{"op":"not_in","value":["manual"]}},"limit":1}}
set LABEL=NOT_IN_TOPLEVEL
set FILE=t61_not_in.json
call :DO_POST

rem ================== 7) Arrays (ARRAY-typed columns; comment out if not present) ==================
> "%OUTDIR%\t70_contains_all.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"tags":{"op":"contains_all","value":["a","b"]}},"limit":1}}
set LABEL=ARRAY_CONTAINS_ALL
set FILE=t70_contains_all.json
call :DO_POST

> "%OUTDIR%\t71_contains_any.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"tags":{"op":"contains_any","value":["a","b"]}},"limit":1}}
set LABEL=ARRAY_CONTAINS_ANY
set FILE=t71_contains_any.json
call :DO_POST

> "%OUTDIR%\t72_contained_by.json" echo {"verb":"db.read","args":{"table":"events","select":["id","topic","source_agent","created_at"],"where":{"tags":{"op":"contained_by","value":["a","b","c"]}},"limit":1}}
set LABEL=ARRAY_CONTAINED_BY
set FILE=t72_contained_by.json
call :DO_POST

echo(
echo =============================================
echo Done. Requests saved under: %OUTDIR%
echo (Each block is a compact summary: ok, rows, meta, error)
echo =============================================
echo(

endlocal
