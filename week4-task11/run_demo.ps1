# PowerShell demo script for Proctoring Hardening

Write-Host "--- Starting FastAPI Server in background ---"
$env:PYTHONPATH="d:\Placemux-aiml\wek4-task11"
Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "api.main:app --host 127.0.0.1 --port 8000"

# Wait for server to start
Start-Sleep -Seconds 5

Write-Host "`n--- 1. Flagged Session Walkthrough ---"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/proctor/check/b5e321cd-8d62-482e-94d1-e147c053b315" | ConvertTo-Json

Write-Host "`n--- 2. Clean Session Walkthrough ---"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/proctor/check/32bd51e0-e6fd-4900-b89f-332fbd52fddb" | ConvertTo-Json

Write-Host "`n--- 3. Edge Cases (Sensor fault, Duplicate, Borderline) ---"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/hardening/edge-cases" | ConvertTo-Json -Depth 5

Write-Host "`n--- 4. Full Metrics Report ---"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/hardening/report" | ConvertTo-Json -Depth 5

Write-Host "`n--- To stop the server, run: Stop-Process -Name uvicorn ---"
