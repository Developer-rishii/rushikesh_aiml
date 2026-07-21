"""
Stage C — Build "An optimised model/serving path meeting the latency SLO".

Approach chosen (see README "Alternative approaches" for what we rejected):
knowledge distillation into a much smaller RandomForest, trained on a blend
of the true labels and the baseline model's predictions (soft targets),
combined with batching + caching in serving.py. This is compared honestly
against the SAME held-out test set the baseline was scored on.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import INTERACTIONS_CSV, ALL_FEATURES, LABEL_COL, RANDOM_SEED
from src.utils import ModelRegistry, log_experiment
from src.evaluate import evaluate_ranking


def main():
    train_df = pd.read_csv(INTERACTIONS_CSV.replace(".csv", "_train.csv"))
    test_df = pd.read_csv(INTERACTIONS_CSV.replace(".csv", "_test.csv"))

    reg = ModelRegistry()
    baseline_model, baseline_entry = reg.load_latest("ranker_baseline")

    X_train = train_df[ALL_FEATURES].values
    y_true = train_df[LABEL_COL].values.astype(float)
    # Soft label = blend of the ground truth and the baseline's own
    # predictions on its training data (classic distillation signal).
    soft_teacher = baseline_model.predict(X_train)
    y_distill = 0.5 * y_true + 0.5 * soft_teacher

    t0 = time.perf_counter()
    small_model = RandomForestRegressor(
        n_estimators=25, max_depth=6, min_samples_leaf=20,
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    small_model.fit(X_train, y_distill)
    train_time_s = time.perf_counter() - t0

    X_test = test_df[ALL_FEATURES].values
    test_df = test_df.copy()
    test_df["score"] = small_model.predict(X_test)
    metrics = evaluate_ranking(test_df, "score", LABEL_COL, "job_id", k=10)
    print("OPTIMIZED (distilled) offline metrics (held-out test):", metrics)

    total_nodes = sum(t.tree_.node_count for t in small_model.estimators_)
    params = {"n_estimators": small_model.n_estimators, "max_depth": 6,
              "total_nodes": int(total_nodes), "train_time_s": round(train_time_s, 2),
              "distilled_from": f"{baseline_entry['name']} v{baseline_entry['version']}"}

    entry = reg.save("ranker_optimized", small_model, metrics, params)
    print("Registered:", entry["name"], "v" + str(entry["version"]),
          f"size={entry['size_bytes']/1e6:.3f} MB "
          f"(baseline was {baseline_entry['size_bytes']/1e6:.2f} MB)")

    log_experiment({"stage": "train_optimized", "metrics": metrics, "params": params,
                     "model_version": entry["version"],
                     "size_reduction_pct": round(
                         100 * (1 - entry["size_bytes"] / baseline_entry["size_bytes"]), 1)})

    return small_model, metrics, params


if __name__ == "__main__":
    main()
