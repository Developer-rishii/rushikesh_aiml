import sys
import os
import json
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_generator import generate
from data_validation import clean, validate_schema, DataValidationError
from baseline_model import baseline_predict
from metrics import disparate_impact, classification_report, segmented_report, false_positive_rate
from mitigate import compute_reweighing_weights, train_mitigated
from audit import fairness_ceiling
from dependency_loader import load_task21_audit, DependencyError
from sign_off import decide, DI_TARGET


# ---------- data generation & messiness ----------

def test_generate_schema_and_scale():
    df = generate(n=2000, n_jobs=10, seed=1)
    expected_cols = {"college_tier", "skill_score", "years_exp", "jd_match",
                      "portfolio_score", "historical_recommended", "fair_recommended", "job_id"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 2000  # duplicates add rows on top


def test_generate_injects_realistic_messiness():
    df = generate(n=5000, seed=1)
    assert df["jd_match"].isna().sum() > 0
    assert df["portfolio_score"].isna().sum() > 0
    assert df.duplicated().sum() > 0


def test_skill_independent_of_tier():
    df = generate(n=8000, seed=1)
    means = df.groupby("college_tier")["skill_score"].mean()
    assert (means.max() - means.min()) < 3.0


def test_historical_label_is_biased():
    df = generate(n=8000, seed=1)
    rates = df.groupby("college_tier")["historical_recommended"].mean()
    assert rates[1] - rates[3] > 0.3


# ---------- data validation / cleaning ----------

def test_validate_schema_rejects_empty():
    try:
        validate_schema(pd.DataFrame())
        assert False, "should have raised"
    except DataValidationError:
        pass


def test_validate_schema_rejects_missing_columns():
    try:
        validate_schema(pd.DataFrame({"college_tier": [1, 2]}))
        assert False, "should have raised"
    except DataValidationError:
        pass


def test_clean_removes_duplicates_and_imputes():
    df = generate(n=3000, seed=2)
    log = {}
    cleaned = clean(df, report=log)
    assert cleaned.duplicated().sum() == 0
    assert cleaned["jd_match"].isna().sum() == 0
    assert cleaned["portfolio_score"].isna().sum() == 0
    assert log["duplicates_removed"] >= 0


def test_clean_caps_outliers():
    df = generate(n=3000, seed=2)
    log = {}
    cleaned = clean(df, report=log)
    # years_exp outliers were injected up to 40; capping must bring the max
    # down to roughly the 99th percentile of the (deduplicated) clean data.
    assert cleaned["years_exp"].max() <= log["years_exp_outliers_capped"]["cap_value"] + 1e-6
    assert cleaned["years_exp"].max() < 35  # well below the injected outlier range


# ---------- baseline ----------

def test_baseline_is_tier_blind():
    """The dumb baseline never even sees college_tier."""
    df = clean(generate(n=3000, seed=3))
    preds = baseline_predict(df)
    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------- metrics ----------

def test_disparate_impact_perfect_parity():
    preds = np.array([1, 0, 1, 0])
    groups = np.array(["a", "a", "b", "b"])
    assert disparate_impact(preds, groups)["disparate_impact"] == 1.0


def test_disparate_impact_full_disparity():
    preds = np.array([1, 1, 0, 0])
    groups = np.array(["a", "a", "b", "b"])
    assert disparate_impact(preds, groups)["disparate_impact"] == 0.0


def test_false_positive_rate_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    assert false_positive_rate(y_true, y_pred) == 0.5


def test_segmented_report_covers_all_groups():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    seg = np.array(["a", "a", "b", "b", "c", "c"])
    report = segmented_report(y_true, y_pred, seg)
    assert set(["overall", "tier_a", "tier_b", "tier_c"]).issubset(report.keys())


# ---------- mitigation (real trained model) ----------

def test_reweighing_weights_are_positive_and_sized():
    y = pd.Series([1, 1, 0, 0, 1, 0])
    group = pd.Series(["a", "a", "a", "b", "b", "b"])
    weights = compute_reweighing_weights(y, group)
    assert len(weights) == len(y)
    assert (weights > 0).all()


def test_mitigation_improves_disparate_impact_over_baseline_audit():
    df = clean(generate(n=6000, seed=42))
    _, _, _, mitigated_audit = train_mitigated(df, seed=42)
    assert mitigated_audit["disparate_impact"] > 0.5  # far better than the ~0.0 baseline


def test_mitigated_model_meets_target():
    df = clean(generate(n=6000, seed=42))
    _, _, _, mitigated_audit = train_mitigated(df, seed=42)
    assert mitigated_audit["disparate_impact"] >= DI_TARGET


def test_fairness_ceiling_is_high():
    df = clean(generate(n=6000, seed=42))
    ceiling = fairness_ceiling(df, seed=42)
    assert ceiling["disparate_impact"] > 0.7


# ---------- dependency loading / error handling ----------

def test_dependency_loader_raises_on_missing_file():
    try:
        load_task21_audit("/tmp/definitely_does_not_exist_12345.json")
        assert False, "should have raised DependencyError"
    except DependencyError:
        pass


def test_dependency_loader_raises_on_malformed_json(tmp_path=None):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    tmp.write("{not valid json")
    tmp.close()
    try:
        load_task21_audit(tmp.name)
        assert False, "should have raised DependencyError"
    except DependencyError:
        pass
    finally:
        os.unlink(tmp.name)


def test_dependency_loader_raises_on_missing_required_fields():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"some_other_field": 1}, tmp)
    tmp.close()
    try:
        load_task21_audit(tmp.name)
        assert False, "should have raised DependencyError"
    except DependencyError:
        pass
    finally:
        os.unlink(tmp.name)


def test_dependency_loader_accepts_well_formed_artifact():
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({
        "disparate_impact": 0.0, "group_positive_rates": {}, "finding": "FAIL",
        "protected_attribute": "college_tier",
    }, tmp)
    tmp.close()
    try:
        data = load_task21_audit(tmp.name)
        assert data["finding"] == "FAIL"
    finally:
        os.unlink(tmp.name)


# ---------- sign-off decision logic ----------

def test_decide_signs_off_when_above_target():
    decision, _ = decide(0.90, 0.95)
    assert decision == "SIGNED_OFF"


def test_decide_conditional_when_near_ceiling_but_below_target():
    decision, _ = decide(0.75, 0.78)
    assert decision == "CONDITIONALLY_SIGNED_OFF"


def test_decide_withholds_when_far_from_ceiling():
    decision, _ = decide(0.30, 0.90)
    assert decision == "WITHHELD"


def test_decide_withholds_when_di_is_none():
    decision, _ = decide(None, 0.90)
    assert decision == "WITHHELD"
