"""
Stage E - Integrate, break it, then demo.
Run: python3 demo.py
This is the ~2 minute live-demo script: runs the full journey end to end on
real (logged) data, shows two tenants behaving differently from identical
code via config only, proves isolation, then deliberately induces the
"model unavailable" failure and shows the designed degradation.
"""
import subprocess
import sys
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


def run(step_name, script):
    print(f"\n{'='*70}\nSTEP: {step_name}\n{'='*70}")
    subprocess.run([sys.executable, script], cwd=SRC, check=True)


if __name__ == "__main__":
    run("1/5 Generate realistic tenant logs (idempotent, seeded)",
        os.path.join(os.path.dirname(SRC), "data", "generate_data.py"))
    run("2/5 Train per-tenant models + evaluate vs baseline (evidence/metrics_report.json)",
        os.path.join(SRC, "train.py"))
    run("3/5 Serve both tenants + induce model-unavailable failure",
        os.path.join(SRC, "serve.py"))
    run("4/5 Prove isolation: access-control + cross-serving + leakage probe",
        os.path.join(SRC, "leakage_test.py"))
    run("5/5 Fairness audit + drift monitor (with simulated shift alert)",
        os.path.join(SRC, "fairness.py"))
    subprocess.run([sys.executable, os.path.join(SRC, "drift.py")], cwd=SRC, check=True)

    print("\nDONE. Evidence written to evidence/*.json and evidence/isolation_proof.txt")
    print("Experiment log: experiments/experiment_log.md")
