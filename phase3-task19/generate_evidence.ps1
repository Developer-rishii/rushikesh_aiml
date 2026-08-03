$workspace = "d:\Placemux-aiml\phase3-task19"
Set-Location $workspace

# Clean data directory
Remove-Item "data\*.pkl" -ErrorAction SilentlyContinue
Remove-Item "data\*.joblib" -ErrorAction SilentlyContinue
Remove-Item "data\*.jsonl" -ErrorAction SilentlyContinue

# Create evidence dir
New-Item -ItemType Directory -Force -Path "evidence" | Out-Null

Set-Location src
Write-Host "=== Running data_gen.py ==="
python data_gen.py 2>&1 | Tee-Object -FilePath "..\evidence\01_data_gen.log"

Write-Host "`n=== Running model.py ==="
python model.py 2>&1 | Tee-Object -FilePath "..\evidence\02_model_train_eval.log"

Write-Host "`n=== Running demo.py ==="
python demo.py 2>&1 | Tee-Object -FilePath "..\evidence\03_full_demo.log"

Write-Host "`n=== Generating fairness guardrail trigger log ==="
python -c @"
import sys, pandas as pd
from policy import PolicyStore
from preview import preview_config
from config import DATA_DIR
scored = pd.read_pickle(DATA_DIR / 'test_scored.pkl')
store = PolicyStore()
# Test with config that triggers fairness guardrails
unfair_overrides = {'w_skill': 1.0, 'w_experience': 0.0, 'w_distance': 0.0}
preview_c, _ = preview_config(store, 'orion_retail', unfair_overrides, scored)
print('orion_retail proposal guardrail_passed:', preview_c['guardrail_passed'])
print('selection rates:', preview_c['fairness_selection_rates'])
for v in preview_c['guardrail_violations']:
    print('  REJECTED:', v)
# Also test with nonsensical config
bad_overrides = {'w_skill': 0.05, 'w_experience': 0.05, 'w_distance': 3.0, 'min_skill_overlap': 0.95}
preview_bad, _ = preview_config(store, 'acme_bank', bad_overrides, scored)
print()
print('acme_bank bad config guardrail_passed:', preview_bad['guardrail_passed'])
for v in preview_bad['guardrail_violations']:
    print('  REJECTED:', v)
"@ 2>&1 | Tee-Object -FilePath "..\evidence\04_fairness_guardrail_trigger.log"

Set-Location ..

Write-Host "`n=== Running automated tests ==="
python tests\test_pipeline.py 2>&1 | Tee-Object -FilePath "evidence\05_automated_tests.log"

Write-Host "`n=== Running train/serve skew check ==="
python tests\train_serve_skew.py 2>&1 | Tee-Object -FilePath "evidence\06_train_serve_skew_check.log"

Write-Host "`n=== ALL DONE ==="
