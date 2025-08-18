#!/usr/bin/env bash
set -euo pipefail

# Enable verbose logs when DEBUG=1 in the workflow env
DEBUG=${DEBUG:-0}
[[ "$DEBUG" == "1" ]] && set -x

: "${DATABASE_URL:?DATABASE_URL is required}"

# --- Normalize DB URL: enforce sslmode=require ---
DBURL="$DATABASE_URL"
if [[ "$DBURL" != *"sslmode="* ]]; then
  DBURL+=$([[ "$DBURL" == *"?"* ]] && echo "&sslmode=require" || echo "?sslmode=require")
fi

# --- Extract hostname and resolve an IPv4 address (prefer A record) ---
DBHOST="$(printf '%s' "$DBURL" | sed -E 's|.*://[^@]*@([^/:?]+).*|\1|')"
PGHOSTADDR="$(getent ahostsv4 "$DBHOST" | awk 'NR==1{print $1}')"
PGHOSTADDR="${PGHOSTADDR:-$DBHOST}"   # fallback if resolution fails
export PGHOSTADDR                     # libpq will prefer this IP

OUT_ROOT=${OUT_ROOT:-schema}                    # location inside the repo
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-public} # comma-separated list (e.g. "public,app")

OUT_DIR="$OUT_ROOT"
TABLE_DIR="$OUT_DIR/tables"
mkdir -p "$TABLE_DIR"

echo "▶ Using host: $DBHOST  (PGHOSTADDR=$PGHOSTADDR)"
echo "▶ Output dir: $OUT_DIR  (tables → $TABLE_DIR)"
echo "▶ Schemas:    $SCHEMAS_TO_INCLUDE"

# Quick connectivity + 10s statement timeout
echo "▶ Checking connectivity…"
psql "$DBURL" -v ON_ERROR_STOP=1 -tAX -c "set statement_timeout = '10s'; select 1;"

# Discover tables
echo "▶ Discovering tables…"
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

echo "▶ Found ${#PAIRS[@]} tables across all app schemas"

# Filter by allowed schemas
in_schemas() {
  local target_csv="$1" line="$2"
  IFS='.' read -r sch _ <<<"$line"
  IFS=',' read -ra want <<<"$target_csv"
  for w in "${want[@]}"; do [[ "$sch" == "$w" ]] && return 0; done
  return 1
}

# Per-table JSON
dumped=0
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  in_schemas "$SCHEMAS_TO_INCLUDE" "$pair" || continue
  IFS='.' read -r sch tbl <<<"$pair"
  out="$TABLE_DIR/${sch}__${tbl}.json"
  echo "  • $sch.$tbl → $out"
  psql "$DBURL" -tAX -v ON_ERROR_STOP=1 \
    -v schema="$sch" -v table="$tbl" \
    -f scripts/schema_per_table.sql > "$out"
  if test -s "$out"; then
    ((dumped++))
  else
    echo "    ↳ warn: empty result for $sch.$tbl" >&2
    rm -f "$out"
  fi
done
echo "▶ Dumped $dumped table files into $TABLE_DIR"

# Build index (tables → column names)
# Protect against empty glob (no table files) so jq doesn't fail with exit 2
shopt -s nullglob
files=( "$TABLE_DIR"/*.json )
if (( ${#files[@]} == 0 )); then
  echo "▶ No per-table JSON files produced; writing empty index."
  printf '{"generated_at":"%s","tables":{}}\n' "$(date -u +%FT%TZ)" > "$OUT_DIR/index.json"
else
  jq -s '{
    generated_at: (now | todateiso8601),
    tables: (map({((.schema + "." + .table)): ([.columns[].name])}) | add)
  }' "${files[@]}" > "$OUT_DIR/index.json"
fi

# Minified index
jq -c . "$OUT_DIR/index.json" > "$OUT_DIR/index.min.json"
echo "▶ Wrote $OUT_DIR/index.json and $OUT_DIR/index.min.json"
