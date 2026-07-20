"""
src/backlog_generator.py
Produces the owned Phase-3 backlog for the matching system.
Items are derived from the defect analysis — ranked by estimated user impact,
not by how interesting the engineering problem is.

Output: reports/phase3_backlog.json + reports/phase3_backlog.md
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT    = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def _load_health_report() -> dict:
    p = REPORTS / "health_report.json"
    if not p.exists():
        raise FileNotFoundError("health_report.json not found — run health_monitor.py first")
    return json.loads(p.read_text())


def _load_defect_result() -> dict:
    p = REPORTS / "experiment_log.jsonl"
    if not p.exists():
        raise FileNotFoundError("experiment_log.jsonl not found — run defect_ranker.py first")
    entries = [json.loads(l) for l in p.read_text().strip().splitlines() if l.strip()]
    return entries[-1]


def _load_ranked_defects() -> pd.DataFrame:
    p = ROOT / "data" / "ranked_defects.csv"
    if not p.exists():
        raise FileNotFoundError("ranked_defects.csv not found — run defect_ranker.py first")
    return pd.read_csv(p)


def generate_backlog() -> dict:
    health    = _load_health_report()
    defect_r  = _load_defect_result()
    ranked    = _load_ranked_defects()

    s         = health["summary"]
    skew_feats = [f for f, v in health["train_serve_skew"].items()
                   if v["skew_detected"]]

    # ── Build backlog items from evidence ─────────────────────────────────────
    items = []

    # Item 1: Train/serve skew (if detected)
    if skew_feats:
        worst_skew_version = max(
            health["by_model_version"].items(),
            key=lambda x: abs(x[1]["mean_skew"])
        )
        items.append({
            "backlog_id":   "B-001",
            "title":        f"Fix train/serve skew in features: {', '.join(skew_feats)}",
            "priority":     "P0",
            "evidence":     (
                f"KS test detects distribution shift in {len(skew_feats)} features. "
                f"Worst affected model version: {worst_skew_version[0]} "
                f"(mean_skew={worst_skew_version[1]['mean_skew']:.3f}). "
                f"Skew accounts for a portion of the "
                f"{abs(s['online_offline_gap']):.3f} online/offline gap."
            ),
            "metric_to_move": "online_offline_gap → 0",
            "affected_users": int(
                ranked[ranked["model_version"].isin(
                    [v for v, vs in health["by_model_version"].items()
                     if abs(vs["mean_skew"]) > 0.03]
                )]["student_id"].nunique()
            ),
            "owner":         "AI/ML + Data Engineering",
            "effort_days":   5,
        })

    # Item 2: Online/offline gap
    gap = abs(s["online_offline_gap"])
    if gap > 0.02:
        items.append({
            "backlog_id":   "B-002",
            "title":        "Investigate online/offline metric gap — model over-confident",
            "priority":     "P0" if gap > 0.05 else "P1",
            "evidence":     (
                f"nDCG@5 offline = {s['ndcg_at_5_offline']:.4f}, "
                f"CTR online = {s['ctr_online']:.4f}, "
                f"gap = {s['online_offline_gap']:.4f} "
                f"({s['gap_direction']}). "
                f"Model scores are systematically biased vs actual user behaviour."
            ),
            "metric_to_move": f"CTR from {s['ctr_online']:.3f} toward {s['expected_ctr_from_score']:.3f}",
            "affected_users": s["total_impressions"],
            "owner":          "AI/ML Engineer",
            "effort_days":    8,
        })

    # Item 3: Top-ranked defects by category
    defect_summary = defect_r.get("defect_summary", {})
    for cat, stats in sorted(defect_summary.items(),
                               key=lambda x: -x[1].get("mean_impact", 0)):
        if cat == "none":
            continue
        bid = f"B-{len(items)+1:03d}"
        items.append({
            "backlog_id":   bid,
            "title":        f"Remediate '{cat}' defects in recommendation ranking",
            "priority":     "P1",
            "evidence":     (
                f"{int(stats['count'])} '{cat}' defects detected, "
                f"mean user impact = {stats['mean_impact']:.3f}. "
                f"Mean defect rank = {stats['mean_rank']:.0f} (lower = hurts more users)."
            ),
            "metric_to_move": f"Defect count → 0 for {cat}",
            "affected_users": int(stats["count"]),
            "owner":          "AI/ML Engineer",
            "effort_days":    6,
        })

    # Item 4: Per-segment worst performer
    worst_tier = None
    worst_ctr  = 1.0
    if "college_tier" in health["by_segment"]:
        for tier, seg in health["by_segment"]["college_tier"].items():
            if seg["ctr"] < worst_ctr:
                worst_ctr  = seg["ctr"]
                worst_tier = tier
    if worst_tier:
        items.append({
            "backlog_id":   f"B-{len(items)+1:03d}",
            "title":        f"Improve matching quality for college_tier={worst_tier} (lowest CTR segment)",
            "priority":     "P1",
            "evidence":     (
                f"college_tier={worst_tier} has CTR={worst_ctr:.4f} — "
                f"the worst-performing segment. This may reflect fairness issues "
                f"carried from the bias audit (Task 21/24)."
            ),
            "metric_to_move": f"Tier {worst_tier} CTR → segment average",
            "affected_users": health["by_segment"]["college_tier"][worst_tier]["n"],
            "owner":          "AI/ML Engineer",
            "effort_days":    10,
        })

    # Item 5: Model versioning / prediction logging completeness
    items.append({
        "backlog_id":   f"B-{len(items)+1:03d}",
        "title":        "Ensure 100% prediction logging with model version + feature snapshot",
        "priority":     "P0",
        "evidence":     (
            "Without complete prediction logging, debugging online/offline gaps "
            "in future sprints is impossible. Currently logging all scored pairs "
            "but feature snapshots at serving time need to be persisted separately "
            "to enable reproducible skew detection."
        ),
        "metric_to_move": "Prediction log completeness → 100%",
        "affected_users": s["total_impressions"],
        "owner":          "AI/ML + Backend",
        "effort_days":    3,
    })

    # Sort by priority then affected_users
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    items.sort(key=lambda x: (priority_order.get(x["priority"], 9),
                               -x["affected_users"]))
    for i, item in enumerate(items):
        item["rank"] = i + 1

    backlog = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "sprint":             "Phase 3 Sprint A — Scale & Reliability",
        "total_items":        len(items),
        "p0_count":           sum(1 for i in items if i["priority"] == "P0"),
        "p1_count":           sum(1 for i in items if i["priority"] == "P1"),
        "key_metrics": {
            "ndcg_at_5":          s["ndcg_at_5_offline"],
            "ctr":                s["ctr_online"],
            "online_offline_gap": s["online_offline_gap"],
            "skew_features":      skew_feats,
            "total_defects":      defect_r["n_defects_in_all_logs"],
        },
        "items": items,
    }

    with open(REPORTS / "phase3_backlog.json", "w") as f:
        json.dump(backlog, f, indent=2, default=str)

    _write_markdown(backlog)
    return backlog


def _write_markdown(backlog: dict) -> None:
    km = backlog["key_metrics"]
    lines = [
        "# Phase 3 Backlog — Matching System",
        f"Generated: {backlog['generated_at'][:19]} UTC",
        f"Sprint: {backlog['sprint']}",
        "",
        "## Key Metrics at Backlog Creation",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| nDCG@5 (offline) | {km['ndcg_at_5']} |",
        f"| CTR (online) | {km['ctr']} |",
        f"| Online/offline gap | {km['online_offline_gap']} |",
        f"| Skewed features | {', '.join(km['skew_features']) or 'None'} |",
        f"| Predicted defects | {km['total_defects']} |",
        "",
        f"## Backlog ({backlog['total_items']} items: "
        f"{backlog['p0_count']} P0, {backlog['p1_count']} P1)",
        "",
        "| Rank | ID | Priority | Title | Affected | Effort |",
        "|------|----|----------|-------|----------|--------|",
    ]
    for item in backlog["items"]:
        lines.append(
            f"| {item['rank']} | {item['backlog_id']} | {item['priority']} | "
            f"{item['title'][:60]} | {item['affected_users']:,} | "
            f"{item['effort_days']}d |"
        )
    lines += ["", "## Item Details", ""]
    for item in backlog["items"]:
        lines += [
            f"### {item['backlog_id']} — {item['title']}",
            f"**Priority:** {item['priority']}  |  "
            f"**Owner:** {item['owner']}  |  "
            f"**Effort:** {item['effort_days']}d",
            f"",
            f"**Evidence:** {item['evidence']}",
            f"",
            f"**Metric to move:** {item['metric_to_move']}",
            f"",
        ]
    with open(REPORTS / "phase3_backlog.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    backlog = generate_backlog()
    print(f"Backlog generated: {backlog['total_items']} items "
          f"({backlog['p0_count']} P0, {backlog['p1_count']} P1)")
    for item in backlog["items"]:
        print(f"  [{item['priority']}] {item['backlog_id']}: {item['title'][:70]}")
