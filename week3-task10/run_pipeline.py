"""
PlaceMux Quality Sign-Off - Pipeline Runner
=============================================
Single entry point: generate data -> build baseline -> train model ->
evaluate -> generate report.

Usage:
    python run_pipeline.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 60)
    print("  PlaceMux Quality Sign-Off - Full Pipeline")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/5] Generating synthetic dataset...")
    from data.generate_dataset import main as gen_data
    gen_data()

    # Step 2: Build baseline
    print("\n[2/5] Computing baseline predictions...")
    from src.baseline import main as run_baseline
    run_baseline()

    # Step 3: Train model
    print("\n[3/5] Training ML model...")
    from src.train_model import main as train
    train()

    # Step 4: Evaluate
    print("\n[4/5] Running evaluation (baseline vs model, pre/post monetization)...")
    from src.evaluation import main as evaluate
    evaluate()

    # Step 5: Generate report
    print("\n[5/5] Generating sign-off report...")
    from reports.generate_report import generate_report
    generate_report()

    print("\n" + "=" * 60)
    print("  [OK] Pipeline complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review: reports/signoff_report.md")
    print("  2. Run tests: python -m pytest tests/ -v")
    print("  3. Start API: uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("  4. Demo:      curl http://localhost:8000/match/S010/J005")


if __name__ == "__main__":
    main()
