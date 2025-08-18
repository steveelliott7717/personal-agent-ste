#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL is required}"

# --- normalize DB URL: add sslmode=require if missing ---
DBURL="$DATABASE_URL"
if [[ "$DBURL" != *"sslmode="* ]]; then
  DBURL+=$([[ "$DBURL" == *"?"* ]] && echo "&sslmode=require" || echo "?sslmode=require")
fi

# --- extract hostname from URI ---
# works for URIs like: postgresql://user:pass@HOST:5432/db?param=...
DBHOST="$(printf '%s' "$DBURL" | sed -E 's|.*://[^@]*@([^/:?]+).*|\1|')"

# --- resolve first IPv4 address for the host ---
PGHOSTADDR="$(getent ahostsv4 "$DBHOST" | awk 'NR==1{print $1}')"
# fallback (shouldn’t happen, but be safe)
PGHOSTADDR="${PGHOSTADDR:-$DBHOST}"

# Export so libpq uses IPv4 even if DNS has AAAA first
export PGHOSTADDR

OUT_ROOT=${OUT_ROOT:-schema}                     # change if you want another folder
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-"public"}   # comma-separated list

OUT_DIR="$OUT_ROOT"
TABLE_DIR="$OUT_DIR/tables"
mkdir -p "$TABLE_DIR"

# Quick connectivity + 10s statement timeout
psql "$DBURL" -v ON_ERROR_STOP=1 -tAX -c "set statement_timeout = '10s'; select 1;"

# Discover tables
readarray -t PAIRS < <(psql "$DBURL" -tAX <<'SQL'
WITH app_schemas AS (
  SELECT nspname AS schema
  FROM pg_namespace
  WHERE nspname NOT IN ('pg_catalog','information_schema','pg_toast')
    AND nspname NOT LIKE 'pg_%'
)
SELECT table_schema || '.' || table_name
FROM information_schema.tables
WHERE table_type='BASE TABLE'
  AND table_schema IN (SELECT schema FROM app_schemas)
ORDER BY table_schema, table_name;
SQL
)

# Filter by allowed schemas
in_schemas() {
  local target_csv="$1" line="$2"
  IFS='.' read -r sch _ <<<"$line"
  IFS=',' read -ra want <<<"$target_csv"
  for w in "${want[@]}"; do [[ "$sch" == "$w" ]] && return 0; done
  return 1
}

# Per-table JSON
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  in_schemas "$SCHEMAS_TO_INCLUDE" "$pair" || continue
  IFS='.' read -r sch tbl <<<"$pair"
  out="$TABLE_DIR/${sch}__${tbl}.json"
  psql "$DBURL" -tAX -v ON_ERROR_STOP=1 \
    -v schema="$sch" -v table="$tbl" \
    -f scripts/schema_per_table.sql > "$out"
  test -s "$out" || { echo "Warn: empty $sch.$tbl" >&2; rm -f "$out"; }
done

# Tiny index (tables -> column names)
jq -s '{
  generated_at: (now | todateiso8601),
  tables: (map({((.schema + "." + .table)): ([.columns[].name])}) | add)
}' "$TABLE_DIR"/*.json > "$OUT_DIR/index.json"

# Minified index
jq -c . "$OUT_DIR/index.json" > "$OUT_DIR/index.min.json"
