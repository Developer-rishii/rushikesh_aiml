"""
run_all.py — runs the entire Task 20 pilot dry-run end-to-end, in order,
exactly as Stage E requires: "Run the full end-to-end journey on real data."

Usage: python3 run_all.py
"""
import subprocess
import sys
import os

STEPS = [
    ("data/generate_data.py", "Generate tenant interaction log (real-shaped data)"),
    ("src/train_ranker.py", "Train ranker on real logged data, held out by job"),
    ("src/evaluate.py", "Offline nDCG/MAP/Precision@10 vs baseline + online proxy"),
    ("src/fairness_audit.py", "Fairness audit: demographic parity + equal opportunity"),
    ("src/latency_bench.py", "Latency benchmark + chaos test (model unavailable)"),
    ("src/explain.py", "Worked explainable example for the demo"),
    ("src/remediation.py", "Generate remediation list from real findings"),
    ("src/model_registry.py", "Write model card for versioning/traceability"),
    ("src/failure_test.py", "Deliberately induce failures, confirm designed degradation"),
]

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    for script, desc in STEPS:
        print(f"\n=== {desc} ({script}) ===")
        result = subprocess.run([sys.executable, script], cwd=root)
        if result.returncode != 0:
            print(f"FAILED at {script}")
            sys.exit(1)
    print("\n=== PILOT DRY-RUN COMPLETE — see experiments/, models/, docs/, demo/ ===")
