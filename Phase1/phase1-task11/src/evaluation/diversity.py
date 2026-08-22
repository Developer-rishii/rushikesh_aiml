"""
evaluation/diversity.py — Step 4: check that diversity (not just
duplication) is driving gains. Directly answers "Do your base models
actually make different mistakes?" with a computed number: pairwise
disagreement rate between base models' predictions on validation data,
and error-overlap (do they get the SAME rows wrong?).
"""
import logging
import itertools
import numpy as np
import pandas as pd

log = logging.getLogger("src.evaluation.diversity")


def compute_pairwise_disagreement(predictions: dict) -> pd.DataFrame:
    """predictions: {model_name: np.array of 0/1 predictions on the same rows}"""
    names = list(predictions.keys())
    rows = []
    for a, b in itertools.combinations(names, 2):
        disagreement_rate = float(np.mean(predictions[a] != predictions[b]))
        rows.append({"model_a": a, "model_b": b, "disagreement_rate": round(disagreement_rate, 4)})
    return pd.DataFrame(rows)


def compute_error_overlap(predictions: dict, y_true) -> dict:
    """For each pair, what fraction of the ROWS EITHER model got wrong
    were gotten wrong by BOTH? Low overlap = diverse errors (good for
    ensembling); high overlap = near-duplicate models (the pitfall)."""
    names = list(predictions.keys())
    y_true = np.asarray(y_true)
    results = {}
    for a, b in itertools.combinations(names, 2):
        wrong_a = predictions[a] != y_true
        wrong_b = predictions[b] != y_true
        either_wrong = wrong_a | wrong_b
        both_wrong = wrong_a & wrong_b
        overlap = float(both_wrong.sum() / either_wrong.sum()) if either_wrong.sum() > 0 else None
        results[f"{a}_vs_{b}"] = {
            "n_wrong_a": int(wrong_a.sum()), "n_wrong_b": int(wrong_b.sum()),
            "n_wrong_both": int(both_wrong.sum()),
            "error_overlap_fraction": round(overlap, 4) if overlap is not None else None,
        }
    return results


def diversity_verdict(overlap_results: dict, threshold: float = 0.9) -> dict:
    """Flags near-identical-model risk: if models that both make errors
    are wrong on the SAME rows >90% of the time, they're not adding
    diverse error-correction — the pitfall "ensembling near-identical
    models" in practice."""
    flagged = {k: v for k, v in overlap_results.items()
               if v["error_overlap_fraction"] is not None and v["error_overlap_fraction"] > threshold}
    verdict = {
        "near_duplicate_pairs_flagged": flagged,
        "diversity_confirmed": len(flagged) == 0,
    }
    log.info("[Step 4] Diversity check: %s", verdict)
    return verdict
