# run_all.ps1 — Windows PowerShell equivalent of run_all.sh
# Stops on first error via $ErrorActionPreference.
$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "========================================"
Write-Host "Phase 3 Task 5: Full Reproduction Run"
Write-Host "========================================"

# Step 1: Generate data
Write-Host ""
Write-Host "[1/8] Generating synthetic data..."
python "$PROJECT_ROOT\data\generate_data.py"

# Step 2: Train model
Write-Host ""
Write-Host "[2/8] Training LightGBM Ranker..."
python "$PROJECT_ROOT\training\train.py"

# Step 3: Fairness evaluation
Write-Host ""
Write-Host "[3/8] Running fairness evaluation..."
python "$PROJECT_ROOT\training\eval_fairness.py"

# Step 4: Explainability
Write-Host ""
Write-Host "[4/8] Generating SHAP explainability report..."
python "$PROJECT_ROOT\training\explain.py"

# Step 5: Start service in background
Write-Host ""
Write-Host "[5/8] Starting service..."
$serviceJob = Start-Job -ScriptBlock {
    param($root)
    Set-Location $root
    python "$root\service\app.py"
} -ArgumentList $PROJECT_ROOT
Start-Sleep -Seconds 4

# Verify service is up
Write-Host "Checking health..."
$health = Invoke-RestMethod http://localhost:8000/health
Write-Host "Health: $($health | ConvertTo-Json)"

# Step 6: Run pytest
Write-Host ""
Write-Host "[6/8] Running pytest..."
Set-Location $PROJECT_ROOT
python -m pytest tests/ -v
Write-Host "All tests passed."

# Step 7: Run load test
Write-Host ""
Write-Host "[7/8] Running load test..."
python "$PROJECT_ROOT\scripts\run_load_test.py"

# Step 8: Failure injection demo
Write-Host ""
Write-Host "[8/8] Running failure injection demo..."
python "$PROJECT_ROOT\scripts\run_failure_demo.py"

# Stop service
Stop-Job $serviceJob -ErrorAction SilentlyContinue
Remove-Job $serviceJob -Force -ErrorAction SilentlyContinue

# Generate sign-off doc from evidence
Write-Host ""
Write-Host "Generating reliability sign-off from evidence..."
python "$PROJECT_ROOT\scripts\generate_signoff.py"

Write-Host ""
Write-Host "========================================"
Write-Host "ALL STEPS COMPLETED SUCCESSFULLY"
Write-Host "========================================"
Write-Host "Evidence files in results/:"
Get-ChildItem "$PROJECT_ROOT\results" | Format-Table Name, Length
Write-Host "Sign-off doc: docs\reliability_sign_off.md"
