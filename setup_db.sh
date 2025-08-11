#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./setup_db.sh                   # apply schema + seed (auto-load .env if present)
#   ./setup_db.sh --schema-only     # only apply schema
#   ./setup_db.sh --no-seed         # apply schema, skip seed
#   ./setup_db.sh --env .env.local  # load from specific env file
#   ./setup_db.sh .env.staging      # (shorthand) same as --env .env.staging
#
# Requires:
#   - psql installed and on PATH (Windows fallback included)
#   - SUPABASE_DB_URL set (via .env or export)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$ROOT_DIR/db"
ENV_FILE=""

DO_SCHEMA=1
DO_SEED=1

# Parse args (allow shorthand: first arg that's a file becomes ENV_FILE)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schema-only) DO_SEED=0 ;;
    --no-seed) DO_SEED=0 ;;
    --env) shift; ENV_FILE="${1:-}";;
    -h|--help)
      sed -n '1,60p' "$0"; exit 0 ;;
    *)
      if [[ -z "$ENV_FILE" && -f "$1" ]]; then
        ENV_FILE="$1"
      else
        echo "Unknown arg: $1" >&2; exit 1
      fi
      ;;
  esac
  shift
done

# Auto-load .env in repo root if present and no env file passed
if [[ -z "${ENV_FILE}" && -f "$ROOT_DIR/.env" ]]; then
  ENV_FILE="$ROOT_DIR/.env"
fi

# Help Git Bash users on Windows find psql automatically
if ! command -v psql >/dev/null 2>&1; then
  for PGBIN in \
    "/c/Program Files/PostgreSQL/17/bin" \
    "/c/Program Files/PostgreSQL/16/bin" \
    "/c/Program Files/PostgreSQL/15/bin"
  do
    if [[ -x "$PGBIN/psql.exe" ]]; then
      export PATH="$PATH:$PGBIN"
      break
    fi
  done
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "Error: psql not found. Install PostgreSQL client and try again." >&2
  echo "Windows tip: add to PATH, e.g. C:\\Program Files\\PostgreSQL\\17\\bin" >&2
  exit 1
fi

# Load env file if provided (robust to comments/blank lines/CRLF)
if [[ -n "${ENV_FILE}" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    # Create a temp sanitized copy: strip CRs and ignore BOM
    TMP_ENV="$(mktemp)"
    tr -d '\r' < "$ENV_FILE" > "$TMP_ENV"
    # Export everything defined in the file (supports quotes)
    set -o allexport
    # shellcheck disable=SC1090
    source "$TMP_ENV"
    set +o allexport
    rm -f "$TMP_ENV"
  else
    echo "Warning: env file '$ENV_FILE' not found. Continuing..." >&2
  fi
fi

if [[ -z "${SUPABASE_DB_URL:-}" ]]; then
  echo "Error: SUPABASE_DB_URL is not set." >&2
  echo "Create a .env file in the repo root or export it in your shell." >&2
  echo "Example .env line:" >&2
  echo 'SUPABASE_DB_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres' >&2
  exit 1
fi

SCHEMA_FILE="$DB_DIR/apply_all.sql"
SEED_FILE="$DB_DIR/seed.sql"

if [[ ! -f "$SCHEMA_FILE" ]]; then
  echo "Error: $SCHEMA_FILE not found. Run from the repo root or fix DB_DIR." >&2
  exit 1
fi

# Print masked connection target
echo "Using database: ${SUPABASE_DB_URL%@*}@${SUPABASE_DB_URL#*@}" | sed 's/:.*@/:***@/' || true

echo "Applying schema from: $SCHEMA_FILE"
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f "$SCHEMA_FILE"

if [[ $DO_SEED -eq 1 ]]; then
  if [[ -f "$SEED_FILE" ]]; then
    echo "Seeding from: $SEED_FILE"
    psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f "$SEED_FILE"
  else
    echo "Seed file not found at $SEED_FILE. Skipping."
  fi
else
  echo "Skipping seed step."
fi

echo "✅ Done."
