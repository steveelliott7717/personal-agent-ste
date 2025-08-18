#!/usr/bin/env bash
set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL is required}"

OUT_ROOT=${OUT_ROOT:-schema}           # change if you want another folder
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-"public"}   # comma-separated list

OUT_DIR="$OUT_ROOT"
TABLE_DIR="$OUT_DIR/tables"
mkdir -p "$TABLE_DIR"

# Keep runs snappy
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -tAX -c "set statement_timeout = '10s';"

# Discover tables
readarray -t PAIRS < <(psql "$DATABASE_URL" -tAX <<'SQL'
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
  psql "$DATABASE_URL" -tAX -v ON_ERROR_STOP=1 \
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
