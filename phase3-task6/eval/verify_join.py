"""
Definition of Done verification: 'Show a real ranked impression and trace
it to the outcome event.'
This is not a claim -- it prints an actual event_id chain pulled from the
real log file so it can be checked by hand.
"""
import pandas as pd
import json
import os

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def trace_one_impression():
    events = pd.read_csv(os.path.join(ARTIFACTS, "event_log.csv"))
    clicks = events[events.event_type == "click"]
    apply_events = events[events.event_type == "apply"]

    # pick an impression that has a full click -> apply chain, if one exists
    applied_ids = set(apply_events.impression_id)
    target_impression_id = next(iter(applied_ids)) if applied_ids else clicks.impression_id.iloc[0]

    impression = events[(events.event_type == "impression") & (events.event_id == target_impression_id)].iloc[0]
    chain = events[
        (events.event_id == target_impression_id) | (events.impression_id == target_impression_id)
    ].sort_values("ts")

    trace = {
        "reconstructed_ranked_list_position": int(impression.position),
        "model_name": impression.model_name,
        "model_version": impression.model_version,
        "query_id": impression.query_id,
        "session_id": impression.session_id,
        "item_id": impression.item_id,
        "full_event_chain": chain[["event_type", "event_id", "position", "model_version", "ts"]].to_dict("records"),
    }
    return trace


def full_join_stats():
    events = pd.read_csv(os.path.join(ARTIFACTS, "event_log.csv"))
    impressions = events[events.event_type == "impression"]
    outcomes = events[events.event_type != "impression"]
    joinable = outcomes.impression_id.isin(impressions.event_id).mean()
    missing_position = impressions.position.isna().mean()
    missing_model_version = impressions.model_version.isna().mean()
    return {
        "total_impressions": int(len(impressions)),
        "total_outcome_events": int(len(outcomes)),
        "outcome_to_impression_joinable_rate": float(joinable),
        "impressions_missing_position": float(missing_position),
        "impressions_missing_model_version": float(missing_model_version),
    }


if __name__ == "__main__":
    result = {"one_traced_impression": trace_one_impression(), "aggregate_join_stats": full_join_stats()}
    with open(os.path.join(ARTIFACTS, "join_verification.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
