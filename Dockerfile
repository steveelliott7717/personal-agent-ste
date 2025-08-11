# syntax=docker/dockerfile:1.6

# --- Stage 1: Build the Vue frontend (Debian base avoids musl quirks) ---
FROM node:20-bullseye AS fe-build
WORKDIR /app/frontend

# Install deps (fallback if npm ci hits optional-deps bug)
COPY frontend/package*.json ./
RUN npm ci || (rm -rf node_modules package-lock.json && npm install)

# Copy sources and build with JS fallback (skip native rollup binary)
COPY frontend/ ./
ENV ROLLUP_SKIP_NODE_BINARY=1
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
