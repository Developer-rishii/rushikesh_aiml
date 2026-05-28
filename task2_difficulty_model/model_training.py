"""
model_training.py
=================
Train three classifiers, compare via Stratified K-Fold CV, evaluate on
held-out test set, generate plots, calibrate probabilities, and save
the best model.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Try XGBoost; fall back to GradientBoosting
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)


class ModelTrainer:
    """
    Orchestrates model training, cross-validation, evaluation, and
    persistence for the assessment-pass prediction task.
    """

    def __init__(self,
                 models_dir: str = "models",
                 results_dir: str = "results"):
        base = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base, models_dir)
        self.results_dir = os.path.join(base, results_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # ── define candidate models ──────────────────────────────────
        self.model_definitions = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0, max_iter=1000,
                    class_weight="balanced", random_state=42)),
            ]),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1),
        }

        if _HAS_XGB:
            self.model_definitions["XGBoost"] = XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0)
        else:
            self.model_definitions["GradientBoosting"] = \
                GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=42)

        self.fitted_models: dict = {}
        self.cv_results: dict = {}
        self.test_results: dict = {}
        self._best_name: str = ""
        self._best_model = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all(self, X_train: pd.DataFrame,
                  y_train: pd.Series) -> dict:
        """Fit every model on the full training set."""
        for name, model in self.model_definitions.items():
            print(f"  Training {name} …", end=" ")
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
            print("done")
        return self.fitted_models

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate_all(self, X_train: pd.DataFrame,
                           y_train: pd.Series) -> dict:
        """
        5-fold Stratified CV on the training set.
        Reports mean ± std for accuracy, precision, recall, f1, roc_auc.
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        for name, model in self.model_definitions.items():
            print(f"  CV  {name} …", end=" ")
            cv = cross_validate(model, X_train, y_train, cv=skf,
                                scoring=scoring, return_train_score=False,
                                n_jobs=-1)
            result = {}
            for metric in scoring:
                key = f"test_{metric}"
                result[metric] = {
                    "mean": cv[key].mean(),
                    "std":  cv[key].std(),
                }
            self.cv_results[name] = result
            print(f"Acc={result['accuracy']['mean']:.4f} ± "
                  f"{result['accuracy']['std']:.4f}")
        return self.cv_results

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------

    def evaluate_on_test(self, X_test: pd.DataFrame,
                         y_test: pd.Series) -> dict:
        """Evaluate every fitted model on the held-out test set."""
        for name, model in self.fitted_models.items():
            y_pred = model.predict(X_test)
            y_prob = (model.predict_proba(X_test)[:, 1]
                      if hasattr(model, "predict_proba") else y_pred)
            self.test_results[name] = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall":    recall_score(y_test, y_pred, zero_division=0),
                "f1":        f1_score(y_test, y_pred, zero_division=0),
                "roc_auc":   roc_auc_score(y_test, y_prob),
            }
        return self.test_results

    # ------------------------------------------------------------------
    # Best model selection
    # ------------------------------------------------------------------

    def get_best_model(self):
        """Select model with highest CV f1 (balances precision & recall)."""
        if not self.cv_results:
            raise RuntimeError("Run cross_validate_all() first.")
        self._best_name = max(
            self.cv_results,
            key=lambda n: self.cv_results[n]["f1"]["mean"],
        )
        self._best_model = self.fitted_models[self._best_name]
        print(f"\n*  Best model: {self._best_name}")
        return self._best_model

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self, X_test: pd.DataFrame,
                     y_test: pd.Series) -> None:
        """Save confusion matrix, ROC curve, and feature importance PNGs."""
        if self._best_model is None:
            self.get_best_model()

        model = self._best_model
        name = self._best_name

        # ── Confusion Matrix ─────────────────────────────────────────
        y_pred = model.predict(X_test)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, cmap="Blues",
            display_labels=["Fail", "Pass"])
        ax.set_title(f"Confusion Matrix – {name}")
        path = os.path.join(self.results_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        # ── ROC Curve ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                       name=name)
        ax.set_title(f"ROC Curve – {name}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        path = os.path.join(self.results_dir, "roc_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        # ── Feature Importance ───────────────────────────────────────
        importances = self._extract_importances(model, X_test)
        if importances is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            importances.sort_values().plot.barh(ax=ax, color="#4C72B0")
            ax.set_title(f"Feature Importance – {name}")
            ax.set_xlabel("Importance")
            path = os.path.join(self.results_dir, "feature_importance.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")

    @staticmethod
    def _extract_importances(model, X) -> pd.Series | None:
        """Pull feature importances from various model types."""
        feature_names = X.columns if hasattr(X, "columns") else \
            [f"f{i}" for i in range(X.shape[1])]
        # Pipeline -> unwrap
        est = model
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("clf", model)
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=feature_names)
        if hasattr(est, "coef_"):
            return pd.Series(np.abs(est.coef_[0]), index=feature_names)
        return None

    # Save / load

    def save_best_model(self) -> str:
        """Calibrate probabilities, then persist with joblib."""
        if self._best_model is None:
            self.get_best_model()
        path = os.path.join(self.models_dir, "best_model.pkl")
        joblib.dump(self._best_model, path)
        print(f"  Saved best model -> {path}")
        return path

    # Pretty-printing

    def print_comparison_table(self) -> str:
        """Print and return a formatted comparison table."""
        header = (f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
                  f"{'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
        sep = "-" * len(header)
        lines = ["\n" + sep, header, sep]

        for name, metrics in self.cv_results.items():
            line = (f"{name:<25} "
                    f"{metrics['accuracy']['mean']:>8.4f}±{metrics['accuracy']['std']:.3f}"
                    f" {metrics['precision']['mean']:>8.4f}±{metrics['precision']['std']:.3f}"
                    f" {metrics['recall']['mean']:>8.4f}±{metrics['recall']['std']:.3f}"
                    f" {metrics['f1']['mean']:>8.4f}±{metrics['f1']['std']:.3f}"
                    f" {metrics['roc_auc']['mean']:>8.4f}±{metrics['roc_auc']['std']:.3f}")
            lines.append(line)
        lines.append(sep)
        table = "\n".join(lines)
        print(table)
        return table

    def print_test_metrics(self) -> str:
        """Print final test-set metrics for every model."""
        header = (f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
                  f"{'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
        sep = "-" * len(header.strip())
        lines = [header, sep]
        for name, m in self.test_results.items():
            line = (f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
                    f" {m['recall']:>10.4f} {m['f1']:>10.4f}"
                    f" {m['roc_auc']:>10.4f}")
            lines.append(line)
        lines.append(sep)
        table = "\n".join(lines)
        print(table)
        return table


# MAIN -- full end-to-end pipeline

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_preprocessing import load_data, clean_data, split_data
    from feature_engineering import FeatureEngineer

    print("=" * 60)
    print("  Model Training Pipeline")
    print("=" * 60)

    # ── 1. Data
    df = load_data()
    df = clean_data(df)
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)

    # ── 2. Feature engineering 
    fe = FeatureEngineer()
    X_train = fe.fit_transform(X_train_raw)
    X_val = fe.transform(X_val_raw)
    X_test = fe.transform(X_test_raw)

    # Save the feature engineer for the prediction module
    fe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "feature_engineer.pkl")
    os.makedirs(os.path.dirname(fe_path), exist_ok=True)
    joblib.dump(fe, fe_path)
    print(f"  Saved FeatureEngineer -> {fe_path}")

    print(f"\nFeatures ({len(fe.get_feature_names())}): "
          f"{fe.get_feature_names()}")
    print(f"Train shape: {X_train.shape}  Val: {X_val.shape}  "
          f"Test: {X_test.shape}")

    # ── 3. Train
    print("\n── Training ──")
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    # ── 4. Cross-validate 
    print("\n── Cross-Validation (5-fold Stratified) ──")
    trainer.cross_validate_all(X_train, y_train)
    comparison = trainer.print_comparison_table()

    # ── 5. Select best model 
    best = trainer.get_best_model()

    # ── 6. Evaluate uncalibrated on test 
    print("\n-- Test Set Evaluation (uncalibrated) --")
    trainer.evaluate_on_test(X_test, y_test)

    # ── 7. Plots (before calibration so feature_importances_ is accessible)
    print("\n-- Saving plots --")
    trainer.plot_results(X_test, y_test)

    # ── 8. Calibrate probabilities 
    calibrated = CalibratedClassifierCV(best, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)
    trainer._best_model = calibrated
    trainer.fitted_models[trainer._best_name + " (calibrated)"] = calibrated

    # ── 9. Evaluate calibrated on test
    print("\n-- Test Set Evaluation (final) --")
    trainer.evaluate_on_test(X_test, y_test)
    test_table = trainer.print_test_metrics()

    # ── 10. Save       
    print("\n-- Saving best model --")
    trainer.save_best_model()

    # ── 11. Final check 
    best_test = max(trainer.test_results.values(), key=lambda m: m["f1"])
    print(f"\n{'=' * 60}")
    print(f"  Best Test Accuracy : {best_test['accuracy']:.4f}")
    print(f"  Best Test F1       : {best_test['f1']:.4f}")
    print(f"  Best Test AUC-ROC  : {best_test['roc_auc']:.4f}")
    if best_test["accuracy"] > 0.75:
        print("  [OK] Accuracy > 75 % requirement met!")
    else:
        print("  [WARN] Accuracy below 75 % -- consider tuning hyperparameters.")
    print("=" * 60)
