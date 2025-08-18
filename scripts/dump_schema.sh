#!/usr/bin/env bash
set -euo pipefail

echo "► Run chmod +x scripts/dump_schema.sh"

# Output directory
OUTDIR="schema"
mkdir -p "$OUTDIR"

# Parse DATABASE_URL
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "❌ DATABASE_URL not set"
  exit 50
fi

# Extract parts
proto="$(echo "$DATABASE_URL" | sed -E 's,^(.*)://.*,\1,')"
user="$(echo "$DATABASE_URL" | sed -E 's,.*//([^:]+):.*,\1,')"
pass="$(echo "$DATABASE_URL" | sed -E 's,.*//[^:]+:([^@]+)@.*,\1,')"
host="$(echo "$DATABASE_URL" | sed -E 's,.*@([^:/]+).*,\1,')"
db="$(echo "$DATABASE_URL" | sed -E 's,.*/([^/?]+).*,\1,')"

# Force pooler host + port
pooler_host="${host/.supabase.co/.pooler.supabase.net}"
port="6543"

echo "► Parsed host: $pooler_host"
echo "► Output dir: $OUTDIR (tables -> $OUTDIR/tables)"
echo "► Schemas: public"

# Connectivity sanity check
echo "► Connectivity sanity (10s timeout)…"
PGPASSWORD="$pass" psql \
  -h "$pooler_host" \
  -p "$port" \
  -U "$user" \
  -d "$db" \
  -c "SELECT now();" \
  --set=sslmode=require \
  --no-align --tuples-only \
  --quiet \
  --connect-timeout=10

if [[ $? -ne 0 ]]; then
  echo "❌ Cannot connect to Postgres with DATABASE_URL (pooler)"
  exit 50
fi

# Dump schema
PGPASSWORD="$pass" pg_dump \
  -h "$pooler_host" \
  -p "$port" \
  -U "$user" \
  -d "$db" \
  --schema-only \
  --no-owner \
  --no-privileges \
  -n public \
  > "$OUTDIR/schema.sql"

echo "✅ Schema dumped to $OUTDIR/schema.sql"
