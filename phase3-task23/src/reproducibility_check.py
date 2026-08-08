import os
import subprocess
import json

BASE = os.path.join(os.path.dirname(__file__), "..")

def run(script_name, args=""):
    cmd = f"python {os.path.join(BASE, 'src', script_name)} {args}"
    print(f"\nRunning: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("=== 1. POPULATING HISTORY (Run 1) ===")
run("generate_data.py", "42")
run("train_model.py")
run("drift_monitor.py")
run("audit_pack.py")

print("=== 2. MULTIPLE SNAPSHOTS (Run 2 - different data seed) ===")
run("generate_data.py", "100")
run("train_model.py")
run("audit_pack.py")

print("=== 3. REPRODUCIBILITY CHECK (Run 3 - identical to Run 2) ===")
run("generate_data.py", "100")
run("train_model.py")

registry_path = os.path.join(BASE, "models", "model_registry.json")
with open(registry_path, "r") as f:
    registry = json.load(f)

# registry has versions for run1, run2, run3
hash_run2 = registry["versions"][-2]["model_hash_sha256_16"]
hash_run3 = registry["versions"][-1]["model_hash_sha256_16"]

print(f"\nModel hash from Run 2 (seed 100): {hash_run2}")
print(f"Model hash from Run 3 (seed 100): {hash_run3}")

if hash_run2 == hash_run3:
    print("SUCCESS: Determinism proven. Hashes match perfectly across identical runs.")
else:
    print("FAIL: Hashes differ.")
    
# Finally run audit pack again so lineage.json has the latest
run("audit_pack.py")
