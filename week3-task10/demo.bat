@echo off
REM PlaceMux Quality Sign-Off — 2-Minute Live Demo Script
REM ========================================================
REM Run these commands in order. Total time: ~2 minutes.

echo [1] Run full pipeline (generate data, train, evaluate)
python run_pipeline.py

echo [2] Run tests
python -m pytest tests/ -v

echo [3] Start API server (background)
start /B uvicorn api.main:app --host 0.0.0.0 --port 8000
timeout /t 3

echo [4] Demo: match explainability
curl http://localhost:8000/match/S010/J005

echo [5] Demo: sign-off report
curl http://localhost:8000/signoff/report

echo [6] Demo: reconciliation
curl http://localhost:8000/signoff/reconciliation
