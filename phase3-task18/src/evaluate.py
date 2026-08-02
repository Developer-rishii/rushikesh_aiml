"""
Stage B/C step 3: "Evaluate on held-out data you did not tune on, and report
the gap between offline metric and expected online effect."

Split is by recruiter (GroupShuffleSplit), not by row -- a row-level split
would leak a recruiter's affinity signal from train into test trivially,
which is exactly the kind of train/serve-skew-adjacent mistake the study
guide calls out as "the single biggest silent killer".
"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_store import ScopedFeatureStore
from src.ranking_model import (build_feature_row, relevance_label,
                                GlobalSignalStub, ndcg_at_k, precision_at_k,
                                average_precision)


def load_events():
    raw = json.loads((Path(__file__).parent.parent / "data" / "interaction_logs.json").read_text())
    return raw["events"], raw["recruiter_org_map"]


def temporal_split_per_recruiter(events, test_frac=0.25, seed=42):
    """First evaluation attempt used GroupShuffleSplit (whole recruiters held
    out). That only measures cold-start on NEVER-SEEN recruiters, which is
    not what "recruiter-scoped personalization" claims to improve -- a
    brand-new recruiter has no scoped history yet by definition, in either
    the scoped or baseline model. Corrected design: for each recruiter, keep
    their chronologically-earlier events in train and later events in test.
    This matches how the system actually runs in production (score today's
    impressions using signal built from this recruiter's past clicks) and is
    the honest test of the scoped-signal hypothesis. Logged here rather than
    silently swapped in, per the "write down WHY, including what you
    rejected" instruction in the study guide."""
    rng = np.random.RandomState(seed)
    by_rec = {}
    for e in events:
        by_rec.setdefault(e["recruiter_id"], []).append(e)
    train, test = [], []
    for rid, evs in by_rec.items():
        order = rng.permutation(len(evs))  # logs aren't timestamped in this
        evs = [evs[i] for i in order]       # synthetic set -> shuffle stands in
        cut = max(1, int(len(evs) * (1 - test_frac)))
        train.extend(evs[:cut])
        test.extend(evs[cut:])
    return train, test


def run_eval(k=10, seed=42, out_path=None):
    events, recruiter_org_map = load_events()
    train_events, test_events = temporal_split_per_recruiter(events, seed=seed)

    # --- build the scoped store from TRAIN events only ---
    store = ScopedFeatureStore()
    for rid, oid in recruiter_org_map.items():
        store.set_membership(rid, oid)
    for e in train_events:
        store.ingest_event(e["recruiter_id"], e["org_id"], e["candidate_skills"], bool(e["clicked"]))

    global_sig = GlobalSignalStub()
    global_sig.fit(train_events)
    store.global_signal = global_sig  # used by build_feature_row(scoped=False)

    results = {}
    for scoped, label in [(True, "scoped_model"), (False, "baseline_global_model")]:
        X_train = np.array([build_feature_row(e, store, scoped) for e in train_events])
        y_train = np.array([relevance_label(e) for e in train_events])
        X_test = np.array([build_feature_row(e, store, scoped) for e in test_events])
        y_test = np.array([relevance_label(e) for e in test_events])

        model = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # group per-recruiter for ranking metrics
        test_groups = {}
        for e, y, p in zip(test_events, y_test, preds):
            test_groups.setdefault(e["recruiter_id"], {"y": [], "p": []})
            test_groups[e["recruiter_id"]]["y"].append(y)
            test_groups[e["recruiter_id"]]["p"].append(p)

        ndcgs, precs, aps = [], [], []
        for g in test_groups.values():
            if len(g["y"]) < 2 or sum(v > 0 for v in g["y"]) == 0:
                continue
            ndcgs.append(ndcg_at_k(g["y"], np.array(g["p"]), k=k))
            precs.append(precision_at_k(g["y"], np.array(g["p"]), k=k))
            aps.append(average_precision(g["y"], g["p"]))

        results[label] = {
            "n_train_events": len(train_events),
            "n_test_events": len(test_events),
            "n_test_recruiters_scored": len(ndcgs),
            "nDCG@10": round(float(np.mean(ndcgs)), 4),
            "MAP": round(float(np.mean(aps)), 4),
            "precision@10": round(float(np.mean(precs)), 4),
        }

    results["lift_scoped_vs_baseline"] = {
        m: round(results["scoped_model"][m] - results["baseline_global_model"][m], 4)
        for m in ["nDCG@10", "MAP", "precision@10"]
    }

    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "artifacts" / "eval_results.json"
    res = run_eval(out_path=out)
    print(json.dumps(res, indent=2))
