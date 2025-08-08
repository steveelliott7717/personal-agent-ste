#!/bin/bash
set -e

# Optional: build the frontend if source exists
if [ -d "./frontend" ]; then
    echo "Building frontend..."
    cd frontend
    if [ -f "package.json" ]; then
        npm install
        npm run build
        # Ensure static dir exists
        mkdir -p ../static
        cp -r dist/* ../static/
    else
        echo "No package.json found in frontend folder — skipping build."
    fi
    cd ..
fi

# Ensure static dir exists (in case no frontend build is done)
mkdir -p static

# Start the backend
echo "Starting backend..."
exec uvicorn main:app --host 0.0.0.0 --port 8080
