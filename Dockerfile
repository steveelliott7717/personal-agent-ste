# ---- Frontend build ----
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy package manifests
COPY frontend/package*.json ./

# Use npm ci if lockfile exists, else fallback to npm install
RUN if [ -f package-lock.json ]; then npm ci --legacy-peer-deps; else npm install --legacy-peer-deps; fi

# Copy frontend source and build
COPY frontend/ .
RUN npm run build


# ---- Backend runtime ----
FROM python:3.11-slim AS backend
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/main.py ./main.py
COPY backend/start.sh ./start.sh
COPY backend/agents ./agents
COPY backend/services ./services
COPY backend/utils ./utils
COPY backend/config ./config

# Copy built frontend into /app/static
COPY --from=frontend-builder /app/frontend/dist ./static

# Make start.sh executable
RUN chmod +x ./start.sh

ENV PORT=8080
EXPOSE 8080
CMD ["./start.sh"]
