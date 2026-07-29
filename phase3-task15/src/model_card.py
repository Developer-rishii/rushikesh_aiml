"""
model_card.py
=============
Generates the model card (Stage D): the governance document that makes a
model auditable and honest about limits (Section 4 definition). Written
fresh per version and stored in governance/model_cards/, not hand-edited,
so it can never drift out of sync with what was actually measured -- one
of the Pitfalls explicitly called out ("A fairness audit done once, at
the end, as a formality" -- here it is regenerated every version, not once).
"""
from pathlib import Path


TEMPLATE = """# Model Card — PlaceMux Candidate Ranker {version}

## Model details
- **Version:** {version}
- **Parent version:** {parent_version}
- **Type:** LightGBM LambdaMART (listwise learning-to-rank), grouped by job_id
- **Training data hash:** `{data_hash}`
- **Feature schema hash:** `{schema_hash}`
- **Trained on:** {train_rows} logged impression rows
- **Registered:** {created_at}

## Intended use
Ranks candidates within a single job's applicant pool so recruiters see the
most likely-to-be-shortlisted candidates first. NOT intended to make
autonomous accept/reject decisions -- output is a ranking signal shown to a
human recruiter, per the DPDP constraint on automated hiring decisions
(Section 3).

## Training data
Real-style logged impressions (impression -> click -> application ->
shortlist funnel), features computed by `src/features.py` (the single
feature-computation layer shared with serving, to prevent train/serve
skew). Protected attribute (`gender`) is excluded from model features and
used only for the fairness audit below.

## Offline metrics (held-out, not tuned on)
| Metric | Baseline (skill_match_score only) | This model | Delta |
|---|---|---|---|
| nDCG@10 | {base_ndcg} | {ndcg} | {delta_ndcg} |
| MAP@10 | {base_map} | {map} | {delta_map} |
| Precision@5 | {base_p5} | {p5} | {delta_p5} |

**Offline-to-online gap:** offline metrics are computed on logged,
already-ranked impressions (position bias exists). This is a KNOWN
limitation, not swept under the rug: Stage E's demo compares this
model's offline win against post-deployment shortlist-rate as the
online proxy, and any model whose offline win doesn't survive that
check is flagged, per the pitfall "shipping an offline win that never
gets validated online."

## Fairness audit ({protected_col}: {groups})
| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| Demographic parity difference | {dpd} | < 0.10 | {dpd_pass} |
| Equal opportunity difference | {eod} | < 0.10 | {eod_pass} |

Selection rate by group: {selection_rate}

**Overall fairness gate: {fairness_pass}**

## Monitoring & rollback
- Drift monitored via PSI on all 6 input features + performance drift on
  nDCG@10 (see `src/drift.py`). Thresholds: PSI alert >= 0.25, performance
  drop >= 8% relative.
- Rollback path: `ModelRegistry.rollback(version)` re-points the
  production pointer to any prior registered version in O(1), fully
  audited in the `promotions` table (who/when/why).
- Failure mode: if the production artifact is unavailable or fails to
  load, `src/serve.py` falls back to `SkillMatchBaseline` and logs a
  `degraded_mode=True` event rather than failing the request.

## Limitations
- Binary relevance label (shortlisted) only; does not yet use graded
  relevance (click < applied < shortlisted), which would give the ranker
  more signal per impression -- deferred, tracked as future work.
- Trained on {train_rows} rows from a single 180-day simulated window;
  seasonal effects beyond that window are unvalidated.
- Fairness audit covers only the `gender` protected attribute recorded in
  this dataset; region-level fairness was not separately gated in v1.

## Who to contact
Governance artifacts hand off to Compliance/DevOps per Section 13; this
card plus the registry DB (`governance/registry.db`) is the complete
audit trail for "which model made a decision on a given date."
"""


def render(version, registry_entry, baseline_metrics, model_metrics, fairness, out_dir="governance/model_cards"):
    import datetime
    m = registry_entry
    def d(a, b):
        return round(b - a, 4)

    txt = TEMPLATE.format(
        version=version,
        parent_version=m["parent_version"] or "—",
        data_hash=m["training_data_hash"],
        schema_hash=m["feature_schema_hash"],
        train_rows=m["train_rows"],
        created_at=datetime.datetime.fromtimestamp(m["created_at"]).isoformat(timespec="seconds"),
        base_ndcg=baseline_metrics["ndcg@10"], ndcg=model_metrics["ndcg@10"], delta_ndcg=d(baseline_metrics["ndcg@10"], model_metrics["ndcg@10"]),
        base_map=baseline_metrics["map@10"], map=model_metrics["map@10"], delta_map=d(baseline_metrics["map@10"], model_metrics["map@10"]),
        base_p5=baseline_metrics["precision@5"], p5=model_metrics["precision@5"], delta_p5=d(baseline_metrics["precision@5"], model_metrics["precision@5"]),
        protected_col="gender", groups=" vs ".join(fairness["groups_compared"]),
        dpd=fairness["demographic_parity_diff"], dpd_pass=fairness["demographic_parity_diff"] < 0.10,
        eod=fairness["equal_opportunity_diff"], eod_pass=(fairness["equal_opportunity_diff"] or 0) < 0.10,
        selection_rate=fairness["selection_rate"],
        fairness_pass=fairness["pass_threshold_0_10"],
    )
    out_path = Path(out_dir) / f"model_card_{version}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt)
    return str(out_path)
