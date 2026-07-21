"""
Stage B — Build "A latency profile of the inference path" starts here: we
first need a real production-shaped model to profile. This trains the
BEFORE model: a deliberately large RandomForest (400 deep trees) of the kind
that is easy to reach for and easy to over-provision, which is exactly the
"model size" bottleneck the study guide's core concepts call out.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import (INTERACTIONS_CSV, ALL_FEATURES, LABEL_COL, RANDOM_SEED)
from src.utils import group_split, log_experiment, ModelRegistry
from src.evaluate import evaluate_ranking


def main():
    df = pd.read_csv(INTERACTIONS_CSV)
    train_df, val_df, test_df = group_split(df, "job_id", seed=RANDOM_SEED)

    X_train, y_train = train_df[ALL_FEATURES].values, train_df[LABEL_COL].values
    X_test, y_test = test_df[ALL_FEATURES].values, test_df[LABEL_COL].values

    print(f"train rows={len(train_df)} (jobs={train_df.job_id.nunique()}) | "
          f"val rows={len(val_df)} | test rows={len(test_df)} (jobs={test_df.job_id.nunique()})")

    t0 = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    test_df = test_df.copy()
    test_df["score"] = model.predict(X_test)
    metrics = evaluate_ranking(test_df, "score", LABEL_COL, "job_id", k=10)
    print("BASELINE offline metrics (held-out test):", metrics)

    n_trees = model.n_estimators
    total_nodes = sum(t.tree_.node_count for t in model.estimators_)
    params = {"n_estimators": n_trees, "max_depth": "None (unbounded)",
              "total_nodes": int(total_nodes), "train_time_s": round(train_time_s, 2)}

    reg = ModelRegistry()
    entry = reg.save("ranker_baseline", model, metrics, params)
    print("Registered:", entry["name"], "v" + str(entry["version"]),
          f"size={entry['size_bytes']/1e6:.2f} MB")

    log_experiment({"stage": "train_baseline", "metrics": metrics, "params": params,
                     "model_version": entry["version"]})

    # Persist test split for downstream profiling/serving/optimization scripts,
    # so every stage evaluates on the exact same held-out data.
    test_df.drop(columns=["score"]).to_csv(
        INTERACTIONS_CSV.replace(".csv", "_test.csv"), index=False)
    train_df.to_csv(INTERACTIONS_CSV.replace(".csv", "_train.csv"), index=False)
    val_df.to_csv(INTERACTIONS_CSV.replace(".csv", "_val.csv"), index=False)

    return model, metrics, params


if __name__ == "__main__":
    main()
