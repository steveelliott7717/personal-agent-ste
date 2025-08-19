#!/usr/bin/env bash
# SCRIPT_VERSION=v3.2
set -uo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

: "${DATABASE_URL:?DATABASE_URL is required}"

OUT_ROOT=${OUT_ROOT:-schema}
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-public}  # e.g. "public,app"
ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}                 # "1" => keep CI green if some tables fail

# Ensure sslmode=require (harmless if already present)
DBURL="$DATABASE_URL"
if [[ "$DBURL" != *"sslmode="* ]]; then
  DBURL+=$([[ "$DBURL" == *"?"* ]] && echo "&sslmode=require" || echo "?sslmode=require")
fi

OUT_DIR="$OUT_ROOT"
TABLE_DIR="$OUT_DIR/tables"
mkdir -p "$TABLE_DIR"

echo "▶ $SCRIPT_VERSION"
echo "▶ Using DATABASE_URL host: $(printf '%s' "$DBURL" | sed -E 's,.*@([^:/]+):.*,\1,')"
echo "▶ Output dir: $OUT_DIR (tables → $TABLE_DIR)"
echo "▶ Schemas:    $SCHEMAS_TO_INCLUDE"

# --- 1) Connectivity (keep -e on for this part)
set -e
echo "▶ Connectivity sanity (PGCONNECT_TIMEOUT=10)…"
PGCONNECT_TIMEOUT=10 psql "$DBURL" -v ON_ERROR_STOP=1 -tAc "select 1;" \
  || { echo "❌ Cannot connect to Postgres with DATABASE_URL"; exit 50; }
echo "1"

# --- 2) Discover tables
echo "▶ Discovering tables…"
DISCOVERY_SQL=$'WITH app_schemas AS (\n  SELECT nspname AS schema\n  FROM pg_namespace\n  WHERE nspname NOT IN (\'pg_catalog\',\'information_schema\',\'pg_toast\')\n    AND nspname NOT LIKE \'pg_%\'\n)\nSELECT table_schema || \'.\' || table_name\nFROM information_schema.tables\nWHERE table_type=\'BASE TABLE\'\n  AND table_schema IN (SELECT schema FROM app_schemas)\nORDER BY table_schema, table_name;'
mapfile -t PAIRS < <(psql "$DBURL" -tAX -c "$DISCOVERY_SQL")
echo "▶ Discovered ${#PAIRS[@]} tables"

# helper: schema filter
in_schemas() {
  local allow="$1" item="$2"
  IFS='.' read -r sch _ <<<"$item"
  IFS=',' read -ra want <<<"$allow"
  for w in "${want[@]}"; do [[ "$sch" == "$w" ]] && return 0; done
  return 1
}

# --- 3) Dump per-table JSON (turn *off* -e to avoid fail-fast)
set +e
echo "▶ Dumping per-table JSON…"
dumped=0
failures=()

for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  in_schemas "$SCHEMAS_TO_INCLUDE" "$pair" || continue

  IFS='.' read -r sch tbl <<<"$pair"
  out="$TABLE_DIR/${sch}__${tbl}.json"
  echo "  • $sch.$tbl → $out"

  errfile="$(mktemp)"
  # ON_ERROR_STOP=1 makes psql return nonzero on any SQL error, but we won't exit due to set +e
  psql "$DBURL" -tAX -v ON_ERROR_STOP=1 -v schema="$sch" -v table="$tbl" \
       -f scripts/schema_per_table.sql >"$out" 2>"$errfile"
  rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "    ❌ FAILED ($rc): $sch.$tbl"
    sed 's/^/      stderr: /' "$errfile" || true
    rm -f "$out"
    failures+=("$sch.$tbl")
  elif [[ ! -s "$out" ]]; then
    echo "    ↳ warn: empty JSON for $sch.$tbl (removed)"
    rm -f "$out"
  else
    ((dumped++))
  fi
  rm -f "$errfile"
done

# restore -e for the rest
set -e

echo "▶ Dumped $dumped table files"
if ((${#failures[@]} > 0)); then
  echo "▶ ${#failures[@]} table(s) failed:"
  for f in "${failures[@]}"; do echo "   - $f"; done
fi

# --- 4) Build compact index
shopt -s nullglob
files=( "$TABLE_DIR"/*.json )
if ((${#files[@]} == 0)); then
  printf '{"generated_at":"%s","tables":{}}\n' "$(date -u +%FT%TZ)" > "$OUT_DIR/index.json"
else
  jq -s '{
    generated_at: (now | todateiso8601),
    tables: (map({((.schema + "." + .table)): ([.columns[].name])}) | add)
  }' "${files[@]}" > "$OUT_DIR/index.json"
fi
jq -c . "$OUT_DIR/index.json" > "$OUT_DIR/index.min.json"
echo "✅ Wrote $OUT_DIR/index.json and $OUT_DIR/index.min.json"

# --- 5) Exit policy
if ((${#failures[@]} > 0)) && [[ "${ALLOW_PARTIAL}" != "1" ]]; then
  echo "❗ Some tables failed. Set ALLOW_PARTIAL=1 to keep the job green."
  exit 1
fi
