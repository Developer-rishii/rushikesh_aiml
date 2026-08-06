"""
extraction_detector.py
Stage D build target: "Detection for scraping/extraction". Model extraction
works by issuing many systematic queries to reconstruct the decision
boundary. Defence = per-client rate + query-diversity (entropy) monitoring;
quotas are the mechanism, entropy/velocity are the signal.
"""
import json, os, math
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

QUERY_RATE_THRESHOLD = 15          # queries per job-window considered high
DIVERSITY_ENTROPY_THRESHOLD = 1.5  # low entropy = repetitive systematic querying


def per_client_stats(interactions):
    by_client = defaultdict(list)
    for row in interactions:
        by_client[row["client_id"]].append(row)

    stats = {}
    for client, rows in by_client.items():
        total_queries = sum(r["query_count"] for r in rows)
        job_counter = Counter(r["job_id"] for r in rows)
        total = sum(job_counter.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in job_counter.values())
        stats[client] = {
            "total_queries": total_queries,
            "distinct_jobs_queried": len(job_counter),
            "query_entropy": entropy,
            "is_scraper_client_ground_truth": any(r["is_scraper_client"] for r in rows),
        }
    return stats


def detect(interactions, out_dir):
    stats = per_client_stats(interactions)
    y_true, y_pred = [], []
    flagged = []
    for client, s in stats.items():
        is_flagged = (s["total_queries"] > QUERY_RATE_THRESHOLD * s["distinct_jobs_queried"] / 5
                      or s["query_entropy"] < DIVERSITY_ENTROPY_THRESHOLD)
        y_true.append(int(s["is_scraper_client_ground_truth"]))
        y_pred.append(int(is_flagged))
        if is_flagged:
            flagged.append(client)

    result = {
        "n_clients": len(stats),
        "n_flagged": len(flagged),
        "flagged_clients": flagged,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "thresholds": {
            "query_rate_threshold": QUERY_RATE_THRESHOLD,
            "diversity_entropy_threshold": DIVERSITY_ENTROPY_THRESHOLD,
        },
        "action_on_flag": "rate-limit to baseline quota + require step-up auth (block, not downrank -- justified in threat_model.md Section 6: false-positive cost for a client is low)",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "extraction_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, stats


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(base, "data", "interactions.json")) as f:
        interactions = json.load(f)
    result, _ = detect(interactions, os.path.dirname(__file__))
    print(json.dumps(result, indent=2))
