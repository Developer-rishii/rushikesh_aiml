"""
Learning-to-rank on org/recruiter-scoped features.

Rejected alternative: global (non-scoped) feature model. Kept as the
explicit BASELINE below so the lift from scoping is measured, not asserted --
the study guide requires "the gap between offline metric and expected online
effect", which requires a baseline to diff against in the first place.

Model choice: sklearn GradientBoostingRegressor as a pointwise ranker
(LightGBM/XGBoost unavailable in this offline environment -- see
DESIGN_DECISIONS.md for the tradeoff). Pointwise on a 0/1/2 relevance label
(click/shortlist/apply) is a documented simplification of listwise LambdaMART;
noted as a "go deeper" follow-up per the study guide.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

SKILLS = ["python", "sql", "react", "node", "aws", "ml", "java", "go",
          "product", "sales", "design", "devops", "data-eng", "qa"]


def relevance_label(row):
    if row["applied"]:
        return 2
    if row["shortlisted"]:
        return 1.5
    if row["clicked"]:
        return 1
    return 0


def skill_overlap_features(candidate_skills, affinity_dict):
    if not affinity_dict:
        return 0.0
    total = sum(affinity_dict.values()) or 1.0
    return sum(affinity_dict.get(s, 0.0) for s in candidate_skills) / total


def build_feature_row(event, store, scoped: bool):
    """scoped=True uses org+recruiter scoped signals (the deliverable).
    scoped=False uses a single global affinity across ALL events, ignoring
    org/recruiter boundaries entirely (the baseline to beat)."""
    cand_skills = event["candidate_skills"]
    skill_onehot = [1.0 if s in cand_skills else 0.0 for s in SKILLS]

    if scoped:
        rec_sig = store.get_recruiter_signal(event["org_id"], event["recruiter_id"])
        org_sig = store.get_org_signal(event["org_id"])
        rec_aff = skill_overlap_features(cand_skills, rec_sig.skill_affinity if rec_sig else {})
        org_aff = skill_overlap_features(cand_skills, org_sig.skill_affinity if org_sig else {})
    else:
        global_sig = store.global_signal
        rec_aff = skill_overlap_features(cand_skills, global_sig.skill_affinity)
        org_aff = rec_aff  # no org distinction in the baseline

    return skill_onehot + [rec_aff, org_aff, event["candidate_years_exp"]]


class GlobalSignalStub:
    """Minimal stand-in exposing the same .skill_affinity shape as
    ScopedSignal, built by pooling ALL orgs together -- deliberately
    reproduces the 'global blob' anti-pattern named in feature_store.py's
    docstring, so the baseline model is genuinely un-scoped."""
    def __init__(self):
        self.skill_affinity = {}

    def fit(self, events):
        from collections import defaultdict
        agg = defaultdict(float)
        for e in events:
            if e["clicked"]:
                for s in e["candidate_skills"]:
                    agg[s] += 1.0
        self.skill_affinity = dict(agg)


def ndcg_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)[:k]
    gains = (2 ** np.array(y_true)[order] - 1)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)
    ideal_order = np.argsort(-np.array(y_true))[:k]
    ideal_gains = (2 ** np.array(y_true)[ideal_order] - 1)
    idcg = np.sum(ideal_gains / discounts[:len(ideal_gains)])
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)[:k]
    return float(np.mean(np.array(y_true)[order] > 0)) if len(order) else 0.0


def average_precision(y_true, y_score):
    order = np.argsort(-np.array(y_score))
    y_true = np.array(y_true)[order]
    hits, ap = 0, 0.0
    for i, y in enumerate(y_true, start=1):
        if y > 0:
            hits += 1
            ap += hits / i
    return ap / hits if hits else 0.0
