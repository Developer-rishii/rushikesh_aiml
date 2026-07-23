"""
run_pipeline.py
-----------------
Stage E: "Integrate, break it, then demo" -- runs the entire pipeline
top-to-bottom exactly once, in order, so every output file in outputs/ is
produced by a single reproducible command: `python run_pipeline.py`
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Stage: generate simulated interaction logs", ["python3", "src/data_generation.py"]),
    ("Stage B: train churn model (time-based split, experiment log)", ["python3", "src/train_model.py"]),
    ("Stage: register model version", ["python3", "src/model_registry.py"]),
    ("Stage C: honest evaluation (PR curve + lift over baseline)", ["python3", "src/evaluate.py"]),
    ("Stage B.4: explainability + worked example", ["python3", "src/explainability.py"]),
    ("Stage D: prioritised at-risk list for growth", ["python3", "src/at_risk_list.py"]),
    ("Fairness audit (run at multiple stages)", ["python3", "fairness/fairness_audit.py"]),
    ("Stage E.3: deliberately break it (model-unavailable failure)", ["python3", "src/failure_simulation.py"]),
]


def main():
    for label, cmd in STEPS:
        print("\n" + "=" * 90)
        print(label)
        print("=" * 90)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[run_pipeline] FAILED at step: {label}")
            sys.exit(1)
    print("\n" + "=" * 90)
    print("ALL STAGES COMPLETE. See outputs/ for every artifact + evidence file.")
    print("=" * 90)


if __name__ == "__main__":
    main()
