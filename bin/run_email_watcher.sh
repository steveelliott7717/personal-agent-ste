#!/usr/bin/env bash
set -euo pipefail
export IMAP_HOST="${IMAP_HOST:-imap.gmail.com}"
export IMAP_PORT="${IMAP_PORT:-993}"
export IMAP_SSL="${IMAP_SSL:-true}"
export IMAP_FOLDER="${IMAP_FOLDER:-INBOX}"
export POLL_SECS="${POLL_SECS:-120}"
python -u backend/workers/email_export_watcher.py
