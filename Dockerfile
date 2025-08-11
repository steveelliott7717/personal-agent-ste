# syntax=docker/dockerfile:1.6

# --- Stage 1: Build the Vue frontend (robust against Rollup optional-deps) ---
FROM node:20-bullseye AS fe-build
WORKDIR /app/frontend

# Install deps from lockfile
COPY frontend/package*.json ./
RUN npm ci

# Copy sources
COPY frontend/ ./

# Ensure the platform-specific rollup package exists; if missing, install the exact matching version.
# Using a heredoc avoids fragile shell quoting.
RUN node - <<'NODE'
const { execSync } = require('node:child_process');
try {
  require('@rollup/rollup-linux-x64-gnu');
  process.exit(0);
} catch (e) {
  const v = require('./node_modules/rollup/package.json').version;
  console.log(`[fix] Installing @rollup/rollup-linux-x64-gnu@${v}`);
  execSync(`npm install --no-save @rollup/rollup-linux-x64-gnu@${v}`, { stdio: 'inherit' });
}
NODE

# (Optional) Also allow JS fallback if native still gets skipped
ENV ROLLUP_SKIP_NODE_BINARY=1

# Build
RUN npm run build

# --- Stage 2: FastAPI runtime serving SPA + API ---
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app
COPY --from=fe-build /app/frontend/dist/ /app/backend/static/

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
