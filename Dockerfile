# ---- Frontend build ----
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Install deps
COPY frontend/package*.json ./
RUN if [ -f package-lock.json ]; then npm ci --legacy-peer-deps; else npm install --legacy-peer-deps; fi

# Build
COPY frontend/ .
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
