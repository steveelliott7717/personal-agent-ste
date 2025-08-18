#!/usr/bin/env bash
set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

: "${DATABASE_URL:?DATABASE_URL is required}"

# 1) Enforce sslmode=require on the URL
DBURL="$DATABASE_URL"
if [[ "$DBURL" != *"sslmode="* ]]; then
  DBURL+=$([[ "$DBURL" == *"?"* ]] && echo "&sslmode=require" || echo "?sslmode=require")
fi

# 2) Parse host safely (strip creds, port, query)
DBNO_SCHEME="${DBURL#*://}"          # drop scheme
AFTER_AT="${DBNO_SCHEME#*@}"         # drop user:pass@ if present
HOSTPORTQ=$([[ "$AFTER_AT" == "$DBNO_SCHEME" ]] && echo "$DBNO_SCHEME" || echo "$AFTER_AT")
HOSTPORT="${HOSTPORTQ%%/*}"          # up to first /
HOSTONLY="${HOSTPORT%%:*}"           # drop :port
HOSTONLY="${HOSTONLY%%\?*}"          # drop ?query

# 3) Prefer IPv4 if present; ONLY export PGHOSTADDR when it's an IP
IPV4_ADDR="$(getent ahostsv4 "$HOSTONLY" | awk 'NR==1{print $1}' || true)"
export PGHOST="$HOSTONLY"
if [[ -n "${IPV4_ADDR:-}" ]]; then
  export PGHOSTADDR="$IPV4_ADDR"
  echo "▶ Parsed host: $HOSTONLY  (using IPv4 PGHOSTADDR=$PGHOSTADDR)"
else
  # No A record found; let libpq resolve (may use AAAA/IPv6)
  unset PGHOSTADDR || true
  echo "▶ Parsed host: $HOSTONLY  (no IPv4 A-record; letting libpq resolve)"
fi

# 4) Output locations
OUT_ROOT=${OUT_ROOT:-schema}
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-public}
OUT_DIR="$OUT_ROOT"
TABLE_DIR="$OUT_DIR/tables"
mkdir -p "$TABLE_DIR"

echo "▶ Output dir: $OUT_DIR (tables → $TABLE_DIR)"
echo "▶ Schemas:    $SCHEMAS_TO_INCLUDE"

# 5) Connectivity sanity (10s)
echo "▶ Connectivity sanity (10s timeout)…"
PGCONNECT_TIMEOUT=10 psql "$DBURL" -v ON_ERROR_STOP=1 -tAX \
  -c "set statement_timeout='10s'; select 1;" \
  || { echo "❌ Cannot connect to Postgres with DATABASE_URL"; exit 50; }

# 6) Discover tables in app schemas
echo "▶ Discovering tables…"
DISCOVERY_SQL=$'WITH app_schemas AS (\n  SELECT nspname AS schema\n  FROM pg_namespace\n  WHERE nspname NOT IN (\'pg_catalog\',\'information_schema\',\'pg_toast\')\n    AND nspname NOT LIKE \'pg_%\'\n)\nSELECT table_schema || \'.\' || table_name\nFROM information_schema.tables\nWHERE table_type=\'BASE TABLE\'\n  AND table_schema IN (SELECT schema FROM app_schemas)\nORDER BY table_schema, table_name;'
mapfile -t PAIRS < <(psql "$DBURL" -tAX -c "$DISCOVERY_SQL")

echo "▶ Discovered ${#PAIRS[@]} tables in app schemas"

# Helper: filter to allowed schemas
in_schemas() {
  local allow="$1" item="$2"
  IFS='.' read -r sch _ <<<"$item"
  IFS=',' read -ra want <<<"$allow"
  for w in "${want[@]}"; do [[ "$sch" == "$w" ]] && return 0; done
  return 1
}

# 7) Per-table JSON dumps
echo "▶ Dumping per-table JSON…"
dumped=0
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  in_schemas "$SCHEMAS_TO_INCLUDE" "$pair" || continue
  IFS='.' read -r sch tbl <<<"$pair"
  out="$TABLE_DIR/${sch}__${tbl}.json"
  echo "  • $sch.$tbl → $out"
  psql "$DBURL" -tAX -v ON_ERROR_STOP=1 -v schema="$sch" -v table="$tbl" \
    -f scripts/schema_per_table.sql > "$out" \
    || { echo "❌ Dump failed for $sch.$tbl" >&2; rm -f "$out"; exit 61; }
  if [[ -s "$out" ]]; then
    ((dumped++))
  else
    echo "⚠️  Empty JSON for $sch.$tbl (removed)" >&2
    rm -f "$out"
  fi
done
echo "▶ Dumped $dumped table files into $TABLE_DIR"

# 8) Build compact index (tables → [columns])
shopt -s nullglob
files=( "$TABLE_DIR"/*.json )
if ((${#files[@]} == 0)); then
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
echo "✅ Wrote $OUT_DIR/index.json and $OUT_DIR/index.min.json"
