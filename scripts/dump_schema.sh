#!/usr/bin/env bash
set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

: "${DATABASE_URL:?DATABASE_URL is required}"

# ── Build a pooled URL from your direct DATABASE_URL ───────────────────────────
# Direct host:  db.<ref>.supabase.co:5432
# Pooled host:  pooler.supabase.com:6543  (shared)
build_pooled_url() {
  local url="$1"
  [[ "$url" == *"://"* ]] || { echo "bad DATABASE_URL"; exit 40; }

  local scheme rest
  scheme="${url%%://*}"
  rest="${url#*://}"

  # split userinfo@host:port/db?query
  local userinfo hostport dbq host port
  if [[ "$rest" == *"@"* ]]; then
    userinfo="${rest%@*}"
    rest="${rest#*@}"
  else
    userinfo=""
  fi
  hostport="${rest%%/*}"
  dbq="${rest#*/}"

  host="${hostport%%:*}"
  port="${hostport#*:}"
  [[ "$port" == "$hostport" ]] && port="5432"

  # Force pooled host + port
  host="pooler.supabase.com"
  port="6543"

  # Reassemble (preserve userinfo if present)
  local prefix
  if [[ -n "$userinfo" ]]; then
    prefix="$scheme://$userinfo@$host:$port"
  else
    prefix="$scheme://$host:$port"
  fi

  local pooled="${prefix}/${dbq}"
  # Ensure sslmode=require
  if [[ "$pooled" != *"sslmode="* ]]; then
    pooled+=$([[ "$pooled" == *"?"* ]] && echo "&sslmode=require" || echo "?sslmode=require")
  fi
  echo "$pooled"
}

POOLED_URL="$(build_pooled_url "$DATABASE_URL")"
echo "▶ Using pooled URL host: $(echo "$POOLED_URL" | sed -E 's,.*@([^:/]+):.*,\1,')"
OUT_ROOT=${OUT_ROOT:-schema}
SCHEMAS_TO_INCLUDE=${SCHEMAS_TO_INCLUDE:-public}
mkdir -p "$OUT_ROOT/tables"

# ── Connectivity sanity (10s) ──────────────────────────────────────────────────
echo "▶ Connectivity sanity (PGCONNECT_TIMEOUT=10)…"
PGCONNECT_TIMEOUT=10 psql "$POOLED_URL" -v ON_ERROR_STOP=1 -tAc "select 1;" \
  || { echo "❌ Cannot connect to Postgres (pooler)"; exit 50; }

# ── Discover tables (exclude system schemas) ───────────────────────────────────
echo "▶ Discovering tables…"
DISCOVERY_SQL=$'WITH app_schemas AS (\n  SELECT nspname AS schema\n  FROM pg_namespace\n  WHERE nspname NOT IN (\'pg_catalog\',\'information_schema\',\'pg_toast\')\n    AND nspname NOT LIKE \'pg_%\'\n)\nSELECT table_schema || \'.\' || table_name\nFROM information_schema.tables\nWHERE table_type=\'BASE TABLE\'\n  AND table_schema IN (SELECT schema FROM app_schemas)\nORDER BY table_schema, table_name;'
mapfile -t PAIRS < <(psql "$POOLED_URL" -tAX -c "$DISCOVERY_SQL")
echo "▶ Discovered ${#PAIRS[@]} tables"

in_schemas() {
  local allow="$1" item="$2"
  IFS='.' read -r sch _ <<<"$item"
  IFS=',' read -ra want <<<"$allow"
  for w in "${want[@]}"; do [[ "$sch" == "$w" ]] && return 0; done
  return 1
}

# ── Per-table JSON dumps ───────────────────────────────────────────────────────
echo "▶ Dumping per-table JSON…"
dumped=0
for pair in "${PAIRS[@]}"; do
  [[ -z "$pair" ]] && continue
  in_schemas "$SCHEMAS_TO_INCLUDE" "$pair" || continue
  IFS='.' read -r sch tbl <<<"$pair"
  out="$OUT_ROOT/tables/${sch}__${tbl}.json"
  echo "  • $sch.$tbl → $out"
  psql "$POOLED_URL" -tAX -v ON_ERROR_STOP=1 -v schema="$sch" -v table="$tbl" \
    -f scripts/schema_per_table.sql > "$out" \
    || { echo "❌ Dump failed for $sch.$tbl" >&2; rm -f "$out"; exit 61; }
  if [[ -s "$out" ]]; then ((dumped++)); else rm -f "$out"; fi
done
echo "▶ Dumped $dumped table files"

# ── Build compact index (tables → [columns]) ───────────────────────────────────
shopt -s nullglob
files=( "$OUT_ROOT/tables"/*.json )
if ((${#files[@]} == 0)); then
  printf '{"generated_at":"%s","tables":{}}\n' "$(date -u +%FT%TZ)" > "$OUT_ROOT/index.json"
else
  jq -s '{
    generated_at: (now | todateiso8601),
    tables: (map({((.schema + "." + .table)): ([.columns[].name])}) | add)
  }' "${files[@]}" > "$OUT_ROOT/index.json"
fi
jq -c . "$OUT_ROOT/index.json" > "$OUT_ROOT/index.min.json"
echo "✅ Wrote $OUT_ROOT/index.json and $OUT_ROOT/index.min.json"
