#!/usr/bin/env bash
set -euo pipefail
# Proton Bridge defaults: IMAP on 1143 (STARTTLS). If you later switch to 993, set IMAP_SSL=true.
export IMAP_HOST="${IMAP_HOST:-127.0.0.1}"
export IMAP_PORT="${IMAP_PORT:-1143}"
export IMAP_SSL="${IMAP_SSL:-false}"
export IMAP_FOLDER="${IMAP_FOLDER:-INBOX}"
export POLL_SECS="${POLL_SECS:-120}"
python -u backend/workers/email_export_watcher.py
