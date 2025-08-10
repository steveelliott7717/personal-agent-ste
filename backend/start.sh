#!/bin/sh
set -e
echo "Starting backend..."
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
