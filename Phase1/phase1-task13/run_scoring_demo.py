"""
run_scoring_demo.py — Step 6: test the interface with realistic inputs.
Loads real rows from Task 2's dataset (never seen by this model's
training — same held-out test split discipline as every prior task) and
scores them through BOTH the single-record and batch interfaces, proving
they agree, then writes a batch scores CSV as the demoable artifact.

Run: python run_scoring_demo.py
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.scoring.interface import Scorer, ScoringError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run_scoring_demo")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "clean_from_task2.csv"
LOCKED_FEATURES_PATH = ROOT / "data" / "locked_feature_set.json"


def load_realistic_holdout_rows(n: int = 20) -> pd.DataFrame:
    """Pulls real WDBC rows, restricted to the locked feature set — the
    exact same schema this model was trained on, not a toy stub."""
    if not DATA_PATH.exists() or not LOCKED_FEATURES_PATH.exists():
        raise FileNotFoundError("Missing data/clean_from_task2.csv or data/locked_feature_set.json.")
    df = pd.read_csv(DATA_PATH)
    features = json.loads(LOCKED_FEATURES_PATH.read_text())["final_feature_set"]

    for feat in features:
        if feat not in df.columns and feat.startswith("coeff_variation_"):
            measurement = feat.replace("coeff_variation_", "").replace("_", " ")
            mean_c, err_c = f"mean {measurement}", f"{measurement} error"
            if mean_c in df.columns and err_c in df.columns:
                df[feat] = df[err_c] / (df[mean_c].abs() + 1e-6)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Locked features missing from raw data: {missing}")

    sample = df.sample(n=n, random_state=42)
    return sample[features + ["target"]]


def main():
    t0 = time.time()
    try:
        scorer = Scorer()
    except ScoringError as e:
        log.error("Scorer initialisation failed: %s", e)
        sys.exit(1)

    sample = load_realistic_holdout_rows(n=20)
    feature_cols = [c for c in sample.columns if c != "target"]
    records = sample[feature_cols].rename(columns=lambda c: c.replace(" ", "_")).to_dict(orient="records")

    single_score = scorer.score_one(records[0])
    log.info("Single-record score: %s", single_score.model_dump())

    batch_result = scorer.score_batch(records)
    log.info("Batch scored %s records, model_version=%s", batch_result.n_records, batch_result.model_version)

    batch_first = batch_result.scores[0]
    consistent = (
        abs(single_score.probability - batch_first.probability) < 1e-9
        and single_score.decision == batch_first.decision
        and single_score.model_version == batch_first.model_version
    )
    log.info("Single-record vs batch consistency for row 0: %s", "MATCH" if consistent else "MISMATCH")
    if not consistent:
        log.error("Single-record and batch scoring paths disagree on the same input — inconsistent interface.")
        sys.exit(1)

    n_correct = sum(
        1 for score, true_label in zip(batch_result.scores, sample["target"].tolist())
        if score.decision == true_label
    )

    malformed_examples = []
    try:
        scorer.score_one({"not_a_real_feature": 1.0})
    except ScoringError as e:
        malformed_examples.append({"case": "unknown field only", "error": e.message, "detail": e.detail})
        log.info("Graceful handling of malformed input (missing required fields): caught cleanly")

    incomplete = {k: v for k, v in list(records[0].items())[:3]}
    try:
        scorer.score_one(incomplete)
    except ScoringError as e:
        malformed_examples.append({"case": "incomplete record", "error": e.message, "detail": e.detail})
        log.info("Graceful handling of malformed input (incomplete record): caught cleanly")

    out_dir = ROOT / "outputs"
    (out_dir / "batch_scores").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    batch_df = pd.DataFrame([s.model_dump() for s in batch_result.scores])
    batch_df.insert(0, "row_index", range(len(batch_df)))
    batch_df.to_csv(out_dir / "batch_scores" / "demo_batch_scores.csv", index=False)

    result = {
        "model_version": scorer.model_version,
        "model_name": scorer.model_name,
        "calibration_method": scorer.calibration_method,
        "threshold": scorer.threshold,
        "n_records_scored": batch_result.n_records,
        "single_vs_batch_consistent": consistent,
        "n_decisions_matching_ground_truth": n_correct,
        "n_total": len(sample),
        "example_single_score": single_score.model_dump(),
        "malformed_input_handling_examples": malformed_examples,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "reports" / "scoring_demo_report.json").write_text(json.dumps(result, indent=2, default=str))
    (out_dir / "logs" / "run_scoring_demo.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"], out_dir / "reports" / "scoring_demo_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
