"""
harness.py — Step 3 of the architecture: the single train/evaluate
harness any model plugs into. This is the one function that wires
data -> features -> model -> evaluate -> experiment log together.

Swapping models = changing `model.name` in configs/config.yaml. Nothing
in this file changes. This is what "config-driven, modular skeleton"
means in practice, not just in name.

Run standalone: python -m src.harness
Run with a different config: python -m src.harness --config path/to/other.yaml
"""
import sys
import argparse
import logging
import time
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.features.build import apply_drop_columns, build_feature_transformer
from src.models.registry import build_model
from src.evaluate.metrics import compute_metrics
from src.evaluate.experiment_log import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.harness")


def run(config_path: Path = None) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    log.info("Loaded config: %s", cfg)

    # ---- data ----
    try:
        df = load_dataframe(cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(df, cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)

    # ---- features ----
    try:
        X_train = apply_drop_columns(X_train, cfg)
        X_val = apply_drop_columns(X_val, cfg)
        transformer = build_feature_transformer(X_train, cfg)
    except ValueError as e:
        log.error("Feature stage failed: %s", e)
        sys.exit(1)

    # ---- model (plug point) ----
    try:
        estimator = build_model(cfg.model_name, cfg.model_params)
        pipeline = Pipeline([("features", transformer), ("model", estimator)])
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        log.error("Model stage failed: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Model training raised an unexpected error: %s", e)
        sys.exit(1)

    # ---- evaluate (unseen validation data only) ----
    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_val)
        y_pred = pipeline.predict(X_val)
        metrics = compute_metrics(y_val, y_proba, y_pred, cfg.metrics)
    except Exception as e:
        log.error("Evaluation stage failed: %s", e)
        sys.exit(1)

    log.info("Validation metrics (%s primary): %s", cfg.primary_metric, metrics)

    # ---- experiment log (Step 5: confirm metrics flow into the log) ----
    split_sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    row = log_run(cfg.experiment_log_path, cfg, metrics, split_sizes)

    # ---- persist model artifact ----
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.model_dir / f"{cfg.model_name}.joblib"
    joblib.dump(pipeline, model_path)
    log.info("Saved model artifact -> %s", model_path)

    result = {
        "config": str(cfg.config_path),
        "model_name": cfg.model_name,
        "metrics": metrics,
        "split_sizes": split_sizes,
        "experiment_log_row": row,
        "model_path": str(model_path),
        "runtime_seconds": round(time.time() - t0, 2),
    }

    cfg.run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = cfg.run_log_dir / f"run_{cfg.model_name}.log"
    with open(run_log_path, "w") as fh:
        fh.write(
            f"model={cfg.model_name} seed={cfg.seed}\n"
            f"split={split_sizes}\n"
            f"metrics={metrics}\n"
            f"runtime_seconds={result['runtime_seconds']}\n"
        )
    log.info("Wrote run log -> %s", run_log_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Task 3 train/eval harness")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to a config YAML (defaults to configs/config.yaml)")
    args = parser.parse_args()
    result = run(Path(args.config) if args.config else None)
    print(result)


if __name__ == "__main__":
    main()
