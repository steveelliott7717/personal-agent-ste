# syntax=docker/dockerfile:1.6

############################
# Stage 1: Build Vue SPA
############################
FROM node:20-bullseye AS fe_build
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./

# Keep Vite/Rollup happy in CI
RUN node - <<'NODE'
const { execSync } = require('node:child_process');
try { require('@rollup/rollup-linux-x64-gnu'); }
catch {
  const v = require('./node_modules/rollup/package.json').version;
  console.log(`[fix] Installing @rollup/rollup-linux-x64-gnu@${v}`);
  execSync(`npm install --no-save @rollup/rollup-linux-x64-gnu@${v}`, { stdio: 'inherit' });
}
NODE

ENV ROLLUP_SKIP_NODE_BINARY=1
# IMPORTANT: ensure Vite outputs to backend/static (base:'/app/')
RUN npm run build


############################
# Stage 2: FastAPI runtime + Playwright (Chromium)
############################
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app \
    # tell Playwright to use a shared, fixed location
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# System deps:
# - git              -> required for repo.patch.apply / commit+push
# - ca-certificates  -> TLS roots
# - curl             -> used by HEALTHCHECK
# - libpq5           -> runtime lib for psycopg if non-binary wheel is used
# - Chromium deps & fonts -> for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl libpq5 \
    libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 \
    libxrandr2 libxshmfence1 libxkbcommon0 libdrm2 libgbm1 libgtk-3-0 \
    libatspi2.0-0 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libcairo2 libpango-1.0-0 libasound2 \
    libnss3 libnspr4 libdbus-1-3 \
    libu2f-udev libvulkan1 \
    fonts-liberation fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Copy backend before installing Python deps
COPY backend/ /app/backend/

# Python deps + Playwright lib
RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
    && pip install --no-cache-dir playwright

# Install Chromium **into /ms-playwright** (accessible to any user)
RUN mkdir -p ${PLAYWRIGHT_BROWSERS_PATH} \
    && python -m playwright install chromium --with-deps

# Bring in built SPA
COPY --from=fe_build /app/frontend/dist/ /app/backend/static/

# Create non-root user and grant access to /app and /ms-playwright
RUN useradd -m appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser ${PLAYWRIGHT_BROWSERS_PATH}
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# If your canonical ASGI app is backend.api:app, feel free to switch the target here.
CMD ["python","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8080"]
