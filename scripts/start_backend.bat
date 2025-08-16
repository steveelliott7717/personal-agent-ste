@echo off
call .venv\Scripts\activate
python -m uvicorn backend.api:app --reload --port 8000
