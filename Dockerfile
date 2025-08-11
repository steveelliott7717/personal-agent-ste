# --- Stage 1: Build the Vue frontend (robust against Rollup native issues) ---
FROM node:20-bullseye AS fe-build
WORKDIR /app/frontend

# Install deps from lockfile
COPY frontend/package*.json ./
RUN npm ci

# Copy sources
COPY frontend/ ./

# Ensure Rollup's platform-specific native package is present.
# If it's missing (npm optional-deps quirk), install the EXACT matching version.
RUN node -e "try{require('@rollup/rollup-linux-x64-gnu');process.exit(0)}catch(e){process.exit(1)}" \
 || sh -lc "ROLLUP_VER=$(node -p \"require('./node_modules/rollup/package.json').version\"); npm install --no-save @rollup/rollup-linux-x64-gnu@${ROLLUP_VER}"

# (Extra guard) allow JS fallback if native still not present for any reason
ENV ROLLUP_SKIP_NODE_BINARY=1

# Build
RUN npm run build


# --- Stage 2: FastAPI runtime serving SPA + API ---
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Python deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# App code
COPY . /app

# Bring built SPA into backend/static (served at /app/)
COPY --from=fe-build /app/frontend/dist/ /app/backend/static/

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
