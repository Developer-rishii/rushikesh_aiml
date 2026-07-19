"""
Standalone re-validation of the LATEST model artifact against held-out
data (days 24-29, never touched during training) plus the online-CTR
proxy. Writes docs/EVAL_REPORT.md so "evaluate honestly against a
baseline" has a persisted, timestamped artifact -- not just stdout.
"""
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_pipeline import compute_features_batch
from src.training.train_model import mean_ndcg, precision_at_k

ROOT = Path(__file__).resolve().parents[2]


def main():
    version = (ROOT / "artifacts" / "models" / "LATEST_VERSION.txt").read_text().strip()
    version_dir = ROOT / "artifacts" / "models" / version
    with open(version_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(version_dir / "metadata.json") as f:
        metadata = json.load(f)

    df = pd.read_csv(ROOT / "data" / "raw" / "interaction_logs.csv")
    test_df = df[df["day_idx"] >= 24].copy()
    X_test = compute_features_batch(test_df)
    test_df["model_score"] = model.predict(X_test)
    test_df["baseline_score"] = test_df["job_popularity"]

    model_ndcg = mean_ndcg(test_df, "model_score", k=10)
    baseline_ndcg = mean_ndcg(test_df, "baseline_score", k=10)
    model_p5 = precision_at_k(test_df, "model_score", k=5)
    baseline_p5 = precision_at_k(test_df, "baseline_score", k=5)

    top1_model = test_df.loc[test_df.groupby("candidate_id")["model_score"].idxmax()]
    top1_base = test_df.loc[test_df.groupby("candidate_id")["baseline_score"].idxmax()]
    ctr_model = float(top1_model["clicked"].mean())
    ctr_base = float(top1_base["clicked"].mean())

    quality_floor = 0.60
    passes_floor = model_ndcg >= quality_floor

    report_lines = [
        "# Evaluation Report — PlaceMux Candidate-Job Ranking Model",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Model version evaluated: `{version}`",
        f"Feature pipeline version: `{metadata['feature_pipeline_version']}`",
        "",
        "## Held-out set",
        f"- {len(test_df):,} impression rows, days 24-29 (never used in training or tuning)",
        f"- Split strategy: {metadata['split_strategy']}",
        "",
        "## Offline metrics (held-out, never tuned on)",
        "| Metric | Model | Baseline (popularity) | Gap |",
        "|---|---|---|---|",
        f"| nDCG@10 | {model_ndcg:.4f} | {baseline_ndcg:.4f} | {model_ndcg-baseline_ndcg:+.4f} |",
        f"| Precision@5 | {model_p5:.4f} | {baseline_p5:.4f} | {model_p5-baseline_p5:+.4f} |",
        "",
        f"Quality-floor SLO (nDCG@10 >= {quality_floor}): "
        f"**{'PASS' if passes_floor else 'FAIL'}** ({model_ndcg:.4f})",
        "",
        "## Simulated online-effect proxy (top-1-shown CTR)",
        f"- Model: {ctr_model:.4f} | Baseline: {ctr_base:.4f} | "
        f"Uplift: {(ctr_model/max(ctr_base,1e-6)-1)*100:+.1f}%",
        "",
        "**Ship gate**: both the offline nDCG win AND the online CTR-proxy win must be positive. "
        f"Offline gap {'positive' if model_ndcg>baseline_ndcg else 'NEGATIVE'}, "
        f"online-proxy gap {'positive' if ctr_model>ctr_base else 'NEGATIVE'} "
        f"-> {'SHIP' if (model_ndcg>baseline_ndcg and ctr_model>ctr_base) else 'DO NOT SHIP'}.",
        "",
        "## Feature importances (used for per-prediction explanations at serve time)",
    ]
    for k, v in metadata["feature_importances"].items():
        report_lines.append(f"- {k}: {v}")

    report_lines += [
        "",
        "## Known gap / rejected alternative",
        "Objective is pointwise regression on graded relevance (0-3), not full pairwise/listwise "
        "LambdaMART, because no network egress was available in this environment to install "
        "LightGBM/XGBoost. `src/training/train_model.py:build_model()` isolates the model class "
        "behind one function so a LambdaMART objective can be swapped in without touching feature "
        "code, evaluation code, or serving code.",
    ]

    out_path = ROOT / "docs" / "EVAL_REPORT.md"
    out_path.write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
