Push-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot\src;" + $env:PYTHONPATH

Write-Host "== Stage 1/5: generating logged impressions (simulated marketplace logs) =="
python data\generate_logs.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

Write-Host "`n== Stage 2/5: estimating position-bias propensities (intervention harvesting) =="
python src\position_bias.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

Write-Host "`n== Stage 3/5: training LTR models + offline evaluation vs heuristic =="
python src\evaluate.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

Write-Host "`n== Stage 4/5: fairness parity + drift monitoring =="
python src\fairness_drift_runner.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

Write-Host "`n== Stage 5/5: failure-mode + regression tests (deliberately induced failures) =="
python tests\test_failure_and_bias.py
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

Write-Host "`nDone. See reports/metrics.json, reports/fairness_drift.json, reports/*.md"
Pop-Location
