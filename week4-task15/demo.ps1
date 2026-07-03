# demo.ps1
Write-Host "--- 1. Generating & Validating Data ---"
python data\generate_sessions.py

Write-Host "`n--- 2. Computing Baseline, Training Model & Tuning Threshold ---"
python src\model.py

Write-Host "`n--- 3. Running Edge Case Tests ---"
pytest tests\test_edge_cases.py -v

Write-Host "`n--- 4. Generating Sign-Off Report ---"
python src\report_generator.py

Write-Host "`n--- 5. Starting API Server (Background) ---"
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "api.main:app --port 8000"

Start-Sleep -Seconds 5

Write-Host "`n--- 6. Hitting Endpoints ---"

Write-Host "`n[GET /fp-reduction/report]"
curl http://localhost:8000/fp-reduction/report

Write-Host "`n`n[GET /fp-reduction/proof]"
curl http://localhost:8000/fp-reduction/proof

Write-Host "`n`n[GET /fp-reduction/edge-cases]"
curl http://localhost:8000/fp-reduction/edge-cases

Write-Host "`n`nStopping API Server..."
Stop-Process -Name "uvicorn" -ErrorAction SilentlyContinue
Write-Host "Demo Complete."
