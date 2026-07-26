"""
test_failure_and_bias.py
=========================
Stage E.3: "Deliberately induce the failure and confirm the designed
degradation actually happens."

Plain assert-based tests with a tiny built-in runner so they work without
pytest/network access (this sandbox has neither). Also collects fine
under pytest if it happens to be installed elsewhere.

Run directly:  python3 tests/test_failure_and_bias.py
"""
import json
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from serve import Ranker  # noqa: E402
from features import assert_no_leakage, build_features  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _sample_candidates():
    df = pd.read_csv(os.path.join(ROOT, "data", "raw_logs.csv"))
    return df[df.job_id == df.job_id.iloc[0]].copy()


def test_normal_serving_uses_ltr_model():
    ranker = Ranker()
    out = ranker.rank(_sample_candidates())
    assert (out.served_by == "ltr_pairwise_corrected").all()


def test_missing_model_file_falls_back_to_heuristic():
    """Deliberately induce failure: point the ranker at a model file that
    does not exist (simulates a registry outage / bad deploy)."""
    sample = _sample_candidates()
    ranker = Ranker(model_path="/tmp/does_not_exist_12345.npy")
    out = ranker.rank(sample)
    assert (out.served_by == "heuristic_fallback").all()
    assert out["score"].notna().all()
    assert len(out) == len(sample)


def test_missing_feature_column_falls_back_to_heuristic():
    """Deliberately induce failure: corrupt the feature store output by
    dropping a required column (simulates an upstream feature-pipeline bug)."""
    broken = _sample_candidates().drop(columns=["embedding_sim"])
    ranker = Ranker()
    out = ranker.rank(broken)
    assert (out.served_by == "heuristic_fallback").all()
    assert len(out) == len(broken)


def test_leakage_guard_trips_on_position_column():
    sample = _sample_candidates()
    feat = build_features(sample)
    feat["position"] = sample["position"].values
    try:
        assert_no_leakage(feat)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "Leakage guard did NOT trip on a leaked 'position' column."


def test_model_beats_heuristic_offline():
    """Regression guard: the chosen model must clear the bar (beat the
    heuristic on nDCG@10 on held-out data) or the pipeline should fail
    loudly rather than ship silently."""
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    heuristic_ndcg = metrics["metrics"]["heuristic (current production)"]["nDCG@10"]
    chosen_ndcg = metrics["metrics"]["pairwise_corrected (CHOSEN model)"]["nDCG@10"]
    assert chosen_ndcg > heuristic_ndcg, "Chosen model did not beat the production heuristic offline -- DO NOT SHIP."


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
