"""
evaluation/effects.py — Step 5: inspect partial dependence / feature
effects for sense-checking. Answers the guide's brainstorming question
"Can you still explain a prediction to a stakeholder?" with an actual
plot, not an assumption that the model is a black box.
"""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

log = logging.getLogger("src.evaluation.effects")


def top_features_by_importance(fitted_pipeline, feature_names, n_top: int) -> list:
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model {type(model).__name__} has no feature_importances_ attribute.")
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
    return [name for name, _ in ranked[:n_top]]


def plot_partial_dependence(fitted_pipeline, X_train, feature_names, top_features, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4 * len(top_features), 4))
    PartialDependenceDisplay.from_estimator(
        fitted_pipeline, X_train, features=top_features, feature_names=feature_names, ax=ax,
    )
    fig.suptitle("Partial Dependence — top features by importance (sense-check)")
    fig.tight_layout()
    path = out_dir / "partial_dependence.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("[Step 5] Saved partial dependence plot for %s -> %s", top_features, path)
    return str(path)
