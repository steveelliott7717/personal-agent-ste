# syntax=docker/dockerfile:1.6

############################
# Stage 1: Build Vue SPA
############################
FROM node:20-bullseye AS fe-build
WORKDIR /app/frontend

# Install deps from lockfile (best cache key)
COPY frontend/package*.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./

# Ensure platform-specific Rollup binary exists (keeps Vite happy on CI)
RUN node - <<'NODE'
const { execSync } = require('node:child_process');
try {
  require('@rollup/rollup-linux-x64-gnu'); // optional native
  process.exit(0);
} catch {
  const v = require('./node_modules/rollup/package.json').version;
  console.log(`[fix] Installing @rollup/rollup-linux-x64-gnu@${v}`);
  execSync(`npm install --no-save @rollup/rollup-linux-x64-gnu@${v}`, { stdio: 'inherit' });
}
NODE

ENV ROLLUP_SKIP_NODE_BINARY=1
# IMPORTANT: your Vite config should emit to ../backend/static with base:'/app/'
RUN npm run build


############################
# Stage 2: FastAPI runtime
############################
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Minimal system deps (TLS, Postgres client libs); keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend app only (avoids sending node_modules into image)
COPY backend/ /app/backend/

# Bring in built SPA -> served by backend at /app/*
COPY --from=fe-build /app/frontend/dist/ /app/backend/static/

# (Optional) Drop privileges
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Basic container health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# Start API
CMD ["python","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8080"]
