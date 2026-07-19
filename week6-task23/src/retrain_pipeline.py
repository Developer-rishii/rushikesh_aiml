"""
src/retrain_pipeline.py

Orchestrates: reference model -> stream monthly batches -> per-batch
drift check -> retrain-on-trigger -> log to Model Registry.

This implements the Task 23 MLOps foundations by using the Model Registry.
"""
import os
import pandas as pd
from datetime import datetime, timezone

from .features import FEATURE_NAMES
from .model import MatchModel
from .drift_detector import compute_drift_report
from .registry import ModelRegistry
from .feature_store import FeatureStore

def run_monitoring_loop(fs: FeatureStore, registry: ModelRegistry, monthly_interactions: dict,
                         train_months: list, stream_months: list,
                         min_precision=0.6, verbose=True):
    """
    train_months : months used to fit the *initial* production model (v1).
    stream_months: months fed one-by-one afterwards, simulating live traffic;
                   each is checked for drift against the training reference
                   and triggers a retrain if PSI crosses the threshold.
    Returns: (history: list[dict], final_model: MatchModel)
    """
    history = []

    # ---- Stage: initial training (reference window) ----
    train_interactions = pd.concat(
        [monthly_interactions[m] for m in train_months], ignore_index=True
    )

    ref_features = fs.get_historical_features(train_interactions)
    model = MatchModel(version="v1")
    model.fit(ref_features, min_precision=min_precision)
    train_metrics = model.evaluate(ref_features)
    
    # Log to registry
    registry.log_model(model, "v1", train_metrics)

    history.append({"month": "+".join(train_months), "event": "initial_train",
                     "model_version": "v1", "metrics": train_metrics, "drift": None})
    if verbose:
        print(f"[train] v1 trained on {len(ref_features)} rows -> {train_metrics}")

    reference_features = ref_features  # drift comparisons always vs this until a retrain updates it
    version_counter = 1

    # ---- Stage: stream months, monitor drift, retrain on trigger ----
    for m in stream_months:
        batch_interactions = monthly_interactions[m]
        batch_features = fs.get_historical_features(batch_interactions)

        drift = compute_drift_report(reference_features, batch_features, FEATURE_NAMES, model=model)
        pre_retrain_metrics = model.evaluate(batch_features)

        event = "monitor_only"
        if drift["trigger_retrain"]:
            version_counter += 1
            new_version = f"v{version_counter}"
            model = MatchModel(version=new_version)
            model.fit(batch_features, min_precision=min_precision)
            
            post_metrics = model.evaluate(batch_features)
            # Log to registry
            registry.log_model(model, new_version, post_metrics)
            
            reference_features = batch_features  # new reference window
            event = "drift_triggered_retrain"

        post_metrics = model.evaluate(batch_features)

        history.append({
            "month": m, "event": event, "model_version": model.version,
            "drift": drift,
            "metrics_before_action": pre_retrain_metrics,
            "metrics_after_action": post_metrics,
        })
        if verbose:
            print(f"[{m}] drift={drift['status']} (max_psi={drift['max_feature_psi']}, "
                  f"pred_psi={drift['prediction_psi']}) -> {event} "
                  f"| precision={post_metrics['precision']} recall={post_metrics['recall']} "
                  f"fpr={post_metrics['false_positive_rate']}")

    return history, model
