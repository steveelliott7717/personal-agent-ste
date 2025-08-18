#!/usr/bin/env bash
set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

: "${DATABASE_URL:?DATABASE_URL is required}"

# Normalize URL: enforce sslmode=require
DBURL="$DATABASE_URL"
if [[ "$DBURL" != *"sslmode="* ]]; then
  DBURL+=$([[ "$DBURL" == *"?"* ]] && echo "&sslmode=require" || echo "?sslmode=require")
fi

# Resolve IPv4 and export PGHOSTADDR so libpq prefers it
DBHOST="$(printf '%s' "$DBURL" | sed -E 's|.*://[^@]*@([^/:?]+).*|\1|')"
PGHOSTADDR="$(getent ahostsv4 "$DBHOST" | awk 'NR==1{print $1}' || true)"
export PGHOSTADDR="${PGHOSTADDR:-$DBHOST}"

OUT_ROOT=${OUT_ROOT:-schema}
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-public}

OUT_DIR="$OUT_ROOT"
TABLE_DIR="$OUT_DIR/tables"
mkdir -p "$TABLE_DIR"

echo "▶ Using DB host: $DBHOST (PGHOSTADDR=$PGHOSTADDR)"
echo "▶ Output dir: $OUT_DIR (tables → $TABLE_DIR)"
echo "▶ Schemas: $SCHEMAS_TO_INCLUDE"

echo "▶ Connectivity sanity (10s timeout)…"
PGCONNECT_TIMEOUT=10 psql "$DBURL" -v ON_ERROR_STOP=1 -tAX -c "set statement_timeout='10s'; select 1;"

echo "▶ Discovering tables…"
DISCOVERY_SQL=$'WITH app_schemas AS (\n  SELECT nspname AS schema\n  FROM pg_namespace\n  WHERE nspname NOT IN (\'pg_catalog\',\'information_schema\',\'pg_toast\')\n    AND nspname NOT LIKE \'pg_%\'\n)\nSELECT table_schema || \'.\' || table_name\nFROM information_schema.tables\nWHERE table_type=\'BASE TABLE\'\n  AND table_schema IN (SELECT schema FROM app_schemas)\nORDER BY table_schema, table_name;'
mapfile -t PAIRS < <(psql "$DBURL" -tAX -c "$DISCOVERY_SQL")

echo "▶ Discovered ${#PAIRS[@]} tables total"
if ((${#PAIRS[@]} == 0)); then
  echo "⚠️  No tables discovered. Check schemas or DB permissions." >&2
fi

in_schemas() {
  local allow="$1" item="$2"
  IFS='.' read -r sch _ <<<"$item"
  IFS=',' read -ra want <<<"$allow"
  for w in "${want[@]}"; do [[ "$sch" == "$w" ]] && return 0; done
  return 1
}

echo "▶ Dumping per-table JSON…"
dumped=0
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  in_schemas "$SCHEMAS_TO_INCLUDE" "$pair" || continue
  IFS='.' read -r sch tbl <<<"$pair"
  out="$TABLE_DIR/${sch}__${tbl}.json"
  echo "  • $sch.$tbl → $out"
  psql "$DBURL" -tAX -v ON_ERROR_STOP=1 -v schema="$sch" -v table="$tbl" \
    -f scripts/schema_per_table.sql > "$out" || { echo "❌ Dump failed for $sch.$tbl" >&2; exit 61; }
  if [[ -s "$out" ]]; then
    ((dumped++))
  else
    echo "⚠️  Empty JSON for $sch.$tbl (removed)" >&2
    rm -f "$out"
  fi
done
echo "▶ Dumped $dumped table files."

echo "▶ Building index…"
shopt -s nullglob
files=( "$TABLE_DIR"/*.json )
if ((${#files[@]} == 0)); then
  printf '{"generated_at":"%s","tables":{}}\n' "$(date -u +%FT%TZ)" > "$OUT_DIR/index.json"
else
  jq -s '{
    generated_at: (now | todateiso8601),
    tables: (map({((.schema + "." + .table)): ([.columns[].name])}) | add)
  }' "${files[@]}" > "$OUT_DIR/index.json" || { echo "❌ jq failed"; exit 70; }
fi

jq -c . "$OUT_DIR/index.json" > "$OUT_DIR/index.min.json"
echo "✅ Wrote $OUT_DIR/index.json and $OUT_DIR/index.min.json"
