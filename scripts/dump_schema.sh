#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-schema}"
OUT_DIR="$OUT_ROOT/tables"
mkdir -p "$OUT_DIR"

echo "Using DATABASE_URL host: $(echo $DATABASE_URL | sed -E 's#postgres(ql)?://([^:]+):([^@]+)@([^:/]+).*#\4#')"
echo "Output dir: $OUT_DIR"
echo "Schemas: ${SCHEMAS_TO_INCLUDE:-public}"

# Connectivity check
PGCONNECT_TIMEOUT=10 psql "$DATABASE_URL" -c "SELECT 1" >/dev/null
echo "Connectivity sanity OK"

# Discover tables
TABLES=$(psql "$DATABASE_URL" -Atc "SELECT schemaname||'.'||tablename FROM pg_tables WHERE schemaname IN ('public');")
echo "Discovered $(echo "$TABLES" | wc -l) tables"

# Dump per-table JSON
for T in $TABLES; do
  echo "Dumping $T -> $OUT_DIR/${T//./__}.json"
  if ! psql "$DATABASE_URL" -Atc "SELECT row_to_json(t) FROM (SELECT * FROM $T) t;" > "$OUT_DIR/${T//./__}.json"; then
    echo "⚠️  Failed dumping $T"
  fi
done

echo "✅ Schema dump complete"
