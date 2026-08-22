"""
evaluation/latency.py — Step 5: balance accuracy gain against added
complexity/latency. Answers "Is the latency cost of the ensemble
acceptable in production?" with a measured number, not an assumption
that ensembles are "obviously" slower or that the cost doesn't matter.
"""
import time
import logging

log = logging.getLogger("src.evaluation.latency")


def measure_inference_latency(fitted_pipeline, X_sample, n_repeats: int = 50) -> dict:
    if len(X_sample) == 0:
        raise ValueError("Cannot measure latency on an empty sample.")
    fitted_pipeline.predict_proba(X_sample)  # warm-up, excluded from timing

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fitted_pipeline.predict_proba(X_sample)
        times.append(time.perf_counter() - t0)

    total_ms = sum(times) * 1000
    mean_ms = total_ms / n_repeats
    per_row_ms = mean_ms / len(X_sample)
    result = {
        "n_repeats": n_repeats,
        "n_rows_per_call": len(X_sample),
        "mean_batch_latency_ms": round(mean_ms, 4),
        "mean_per_row_latency_ms": round(per_row_ms, 6),
    }
    log.info("[Step 5] Latency: %s", result)
    return result
