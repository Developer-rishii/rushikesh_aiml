Write-Host "=========================================="
Write-Host "  Parsing v0 Live Demo"
Write-Host "=========================================="
Write-Host ""

Write-Host "1. Parsing a fresh, live pasted resume..." -ForegroundColor Cyan
$json1 = @{
    text = "Full-stack dev with 3 yrs experience. Really love working with React.js and Spring Boot. I have no experience with GraphQL unfortunately. Currently playing around with some ML prototypes."
} | ConvertTo-Json
$response1 = Invoke-RestMethod -Uri "http://localhost:8000/parse" -Method Post -Body $json1 -ContentType "application/json"
$response1 | ConvertTo-Json -Depth 5
Write-Host ""

Write-Host "2. Explainability Walkthrough (Hits & Misses on a pre-built example)..." -ForegroundColor Cyan
$response2 = Invoke-RestMethod -Uri "http://localhost:8000/parse/eval/res_alias_01" -Method Get
$response2 | ConvertTo-Json -Depth 5
Write-Host ""

Write-Host "3. Edge Case: Empty/Malformed Input..." -ForegroundColor Cyan
$json3 = @{
    text = "@#$!@#$   "
} | ConvertTo-Json
$response3 = Invoke-RestMethod -Uri "http://localhost:8000/parse" -Method Post -Body $json3 -ContentType "application/json"
$response3 | ConvertTo-Json -Depth 5
Write-Host ""

Write-Host "4. Edge Case: Negation..." -ForegroundColor Cyan
$json4 = @{
    text = "I am not familiar with Azure, but I have used AWS before."
} | ConvertTo-Json
$response4 = Invoke-RestMethod -Uri "http://localhost:8000/parse" -Method Post -Body $json4 -ContentType "application/json"
$response4 | ConvertTo-Json -Depth 5
Write-Host ""

Write-Host "5. Getting Full Metrics Report (Parsing v0 vs Baseline vs Rules-Only)..." -ForegroundColor Cyan
$response5 = Invoke-RestMethod -Uri "http://localhost:8000/parse/report" -Method Get
Write-Host "Overall Precision (Full Pipeline): $($response5.full_pipeline.overall.precision)"
Write-Host "Overall Recall (Full Pipeline): $($response5.full_pipeline.overall.recall)"
Write-Host "Overall Precision (Baseline): $($response5.baseline.overall.precision)"
Write-Host ""
Write-Host "Demo Complete!" -ForegroundColor Green
