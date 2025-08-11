# --- Stage 1: Build the Vue frontend (Debian base avoids musl quirks) ---
FROM node:20-bullseye AS fe-build
WORKDIR /app/frontend

# Install deps
COPY frontend/package*.json ./
# Work around npm optional-deps issue cleanly:
# 1) Try npm ci; if it fails on optional deps, fall back to npm install without lock.
RUN npm ci || (rm -rf node_modules package-lock.json && npm install)

# Copy sources
COPY frontend/ ./

# Force Rollup to skip native binary and use JS fallback
ENV ROLLUP_SKIP_NODE_BINARY=1

# Build
RUN npm run build



# ---- Backend runtime ----
FROM python:3.11-slim AS backend
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend AS A PACKAGE
COPY backend ./backend

# Place built frontend where backend expects it
COPY --from=frontend-builder /app/frontend/dist ./backend/static

# Optional start script
COPY backend/start.sh ./start.sh
RUN chmod +x ./start.sh

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080

# Start the app from the backend package
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
