"""
Step 5 of the build pipeline: a trivial 'fit a dummy model' smoke test,
run end-to-end, with metrics logged via experiment_tracker.

A DummyClassifier (most_frequent strategy) is used deliberately: Day 1's
job is to prove the *pipeline* works (ingest -> split -> train -> log),
not to prove a good model. This also gives a baseline number that any
real model must beat later.
"""
import sys
import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import SEED, TRAIN_PATH, VAL_PATH, TEST_PATH, TARGET_COL, MODEL_DIR
from src.experiment_tracker import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("smoke_test")


def _load_split(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run data_split.py first.")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def run_smoke_test():
    X_train, y_train = _load_split(TRAIN_PATH)
    X_val, y_val = _load_split(VAL_PATH)

    model = DummyClassifier(strategy="most_frequent", random_state=SEED)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    metrics = {
        "val_accuracy": round(accuracy_score(y_val, val_preds), 4),
        "val_f1": round(f1_score(y_val, val_preds), 4),
    }
    log.info("Smoke-test baseline metrics: %s", metrics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "dummy_baseline.joblib"
    joblib.dump(model, model_path)
    log.info("Saved baseline model to %s", model_path)

    log_run(
        run_name="smoke_test_dummy_baseline",
        params={"model": "DummyClassifier", "strategy": "most_frequent", "seed": SEED},
        metrics=metrics,
    )
    return metrics


if __name__ == "__main__":
    run_smoke_test()
