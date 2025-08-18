#!/usr/bin/env bash
set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

: "${DATABASE_URL:?DATABASE_URL is required}"

# 1) Build a pooled URL from the direct DATABASE_URL
#    db.<ref>.supabase.co:5432  ->  db.<ref>.pooler.supabase.net:6543
build_pooled_url() {
  local url="$1"
  # ensure scheme present
  [[ "$url" != *"://"* ]] && { echo "bad DATABASE_URL"; exit 40; }

  # split at '://'
  local scheme rest
  scheme="${url%%://*}"; rest="${url#*://}"

  # userinfo@host:port/db?q
  local userinfo hostport dbq host port
  userinfo="${rest%@*}"; rest="${rest#*@}"             # user:pass (ignored), rest starts at host:port/db?...
  [[ "$userinfo" == "$rest" ]] && rest="${url#*://}"   # no creds case

  hostport="${rest%%/*}"                               # host:port
  dbq="${rest#*/}"                                     # db?query

  host="${hostport%%:*}"
  port="${hostport#*:}"
  [[ "$port" == "$hostport" ]] && port="5432"

  # rewrite host + port
  host="${host/.supabase.co/.pooler.supabase.net}"
  port="6543"

  # reassemble; preserve original userinfo if there was one
  local prefix
  if [[ "$url" == *"@"* ]]; then
    prefix="$scheme://$userinfo@$host:$port"
  else
    # re-extract userinfo from whole URL to be safe
    local ui
    ui="$(echo "$url" | sed -E 's,^[^:]+://([^@]+)@.*,\1,')" || true
    if [[ "$ui" != "$url" ]]; then
      prefix="$scheme://$ui@$host:$port"
    else
      prefix="$scheme://$host:$port"
    fi
  fi

  local pooled="${prefix}/${dbq}"
  # add sslmode=require if missing
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

# 2) Connectivity sanity (10s) — use libpq env vars (no --connect-timeout)
echo "▶ Connectivity sanity (PGCONNECT_TIMEOUT=10)…"
PGCONNECT_TIMEOUT=10 psql "$POOLED_URL" -v ON_ERROR_STOP=1 -tAc "select 1;" \
  || { echo "❌ Cannot connect to Postgres (pooler)"; exit 50; }

# 3) Discover tables
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

# 4) Per-table JSON dumps
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

# 5) Build compact index
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
