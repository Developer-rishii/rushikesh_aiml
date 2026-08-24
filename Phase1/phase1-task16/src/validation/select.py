"""
validation/select.py — Step 6: conclude which model generalises best.
"Most generalising" = highest mean AND acceptably low variance across
folds — a high-mean, high-variance model is inconsistent, not
generalising, per the brainstorming question "Which model wins
consistently, not just on average?"
"""
import logging

log = logging.getLogger("src.validation.select")


def select_most_generalising(per_model_results: dict, cfg) -> dict:
    candidates = []
    for name, res in per_model_results.items():
        acceptable = res["std"] <= cfg.max_acceptable_std
        candidates.append({"model": name, "mean": res["mean"], "std": res["std"], "acceptable_variance": acceptable})

    stable_candidates = [c for c in candidates if c["acceptable_variance"]]
    if stable_candidates:
        winner = max(stable_candidates, key=lambda c: c["mean"])
        fallback_used = False
    else:
        winner = min(candidates, key=lambda c: c["std"])
        fallback_used = True

    consistent_winner_by_min_fold = max(per_model_results.items(), key=lambda kv: kv[1]["min"])[0]

    result = {
        "all_candidates": candidates,
        "selected_model": winner["model"],
        "selected_mean": winner["mean"],
        "selected_std": winner["std"],
        "selection_rule": (
            "highest mean among models with std <= max_acceptable_std"
            if not fallback_used else
            "NO model had acceptable variance; fell back to lowest-std (most consistent) model"
        ),
        "which_model_wins_by_worst_fold": consistent_winner_by_min_fold,
    }
    log.info("[Step 6] Selected most generalising model: %s (mean=%.4f, std=%.4f) | rule: %s",
              result["selected_model"], result["selected_mean"], result["selected_std"], result["selection_rule"])
    return result
