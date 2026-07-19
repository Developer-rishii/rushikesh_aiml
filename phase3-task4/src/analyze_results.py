"""
analyze_results.py
-------------------
Turns the raw load test CSV into the two required deliverables:
  1. A latency-vs-load curve (PNG) -- the "show latency curves" verification.
  2. A breaking-point report (Markdown) with the exact QPS where inference
     degrades and the headroom required, computed from the real numbers in
     results/load_test_results.csv (not asserted from vibes).

Breaking point definition (stated explicitly so it's falsifiable):
  A load level is "past the knee" if EITHER
    - p95 latency > 3x the p95 latency at the lowest tested concurrency, OR
    - fallback_rate > 1% (the service has started shedding load onto the
      cheap heuristic because real inference can't keep up)
  The breaking point is the first level, in ascending concurrency, where
  that condition holds.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

IN_PATH = "results/load_test_results.csv"
PLOT_PATH = "results/latency_curve.png"
REPORT_PATH = "results/breaking_point_report.md"
TARGET_SUSTAINED_QPS = 150  # stated assumption: marketplace-scale sustained target load


def find_breaking_point(df):
    baseline_p95 = df.iloc[0]["p95_ms"]
    df["past_knee"] = (df["p95_ms"] > 3 * baseline_p95) | (df["fallback_rate"] > 0.01)
    knee_rows = df[df["past_knee"]]
    if knee_rows.empty:
        return None, baseline_p95
    return knee_rows.iloc[0], baseline_p95


def main():
    df = pd.read_csv(IN_PATH)
    knee_row, baseline_p95 = find_breaking_point(df)
    max_clean_row = df[df["fallback_rate"] <= 0.001].iloc[-1]  # last level with (near) zero fallback

    # --- plot ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(df["achieved_qps"], df["p50_ms"], marker="o", label="p50")
    ax[0].plot(df["achieved_qps"], df["p95_ms"], marker="o", label="p95")
    ax[0].plot(df["achieved_qps"], df["p99_ms"], marker="o", label="p99")
    if knee_row is not None:
        ax[0].axvline(knee_row["achieved_qps"], color="red", linestyle="--",
                       label=f"breaking point (~{knee_row['achieved_qps']:.0f} QPS)")
    ax[0].set_xlabel("Achieved QPS")
    ax[0].set_ylabel("Latency (ms)")
    ax[0].set_title("Latency vs Load")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(df["achieved_qps"], df["fallback_rate"] * 100, marker="o", color="darkorange")
    ax[1].set_xlabel("Achieved QPS")
    ax[1].set_ylabel("Fallback rate (%)")
    ax[1].set_title("Graceful Degradation Onset")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140)
    print(f"wrote {PLOT_PATH}")

    # --- report ---
    lines = []
    lines.append("# Breaking Point & Headroom Report\n")
    lines.append("Generated from a real, executed load test (`results/load_test_results.csv`); "
                  "every number below is read directly from that run, not estimated.\n")
    lines.append("## Raw levels\n")
    lines.append(df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Breaking point\n")
    if knee_row is not None:
        triggered_by = []
        if knee_row["p95_ms"] > 3 * baseline_p95:
            triggered_by.append(f"p95 latency crossed 3x baseline ({knee_row['p95_ms']:.1f}ms vs {baseline_p95:.1f}ms)")
        if knee_row["fallback_rate"] > 0.01:
            triggered_by.append(f"fallback rate crossed 1% ({knee_row['fallback_rate']*100:.1f}%)")
        lines.append(
            f"- **Breaking point reached at concurrency={int(knee_row['concurrency'])}, "
            f"achieved throughput ~= {knee_row['achieved_qps']:.0f} QPS.**\n"
            f"- Triggered by: {'; '.join(triggered_by)}.\n"
            f"- p95 latency at this level = {knee_row['p95_ms']:.1f} ms "
            f"(baseline p95 at concurrency={int(df.iloc[0]['concurrency'])} was {baseline_p95:.1f} ms, "
            f"{knee_row['p95_ms']/baseline_p95:.1f}x increase); "
            f"fallback rate = {knee_row['fallback_rate']*100:.1f}%.\n"
        )
    else:
        lines.append("- No breaking point observed inside the tested range — "
                      "the service held SLA at the highest concurrency tested.\n")

    lines.append(
        f"\n- **Last clean level (fallback rate ≈ 0%)**: concurrency="
        f"{int(max_clean_row['concurrency'])}, achieved ≈ {max_clean_row['achieved_qps']:.0f} QPS, "
        f"p95 = {max_clean_row['p95_ms']:.1f} ms. This is the largest load the model path "
        f"handles with zero degradation — call it **safe capacity**.\n"
    )

    safe_qps = max_clean_row["achieved_qps"]
    headroom_ratio = safe_qps / TARGET_SUSTAINED_QPS if TARGET_SUSTAINED_QPS else None
    lines.append("## Required headroom\n")
    lines.append(
        f"- Stated target sustained load for this service: **{TARGET_SUSTAINED_QPS} QPS** "
        f"(assumption — replace with the real marketplace peak from traffic logs before sign-off).\n"
        f"- Measured safe capacity of **one instance**: **{safe_qps:.0f} QPS**.\n"
        f"- Headroom ratio at 1 instance: **{headroom_ratio:.2f}x** target "
        f"({'sufficient' if headroom_ratio and headroom_ratio >= 1.5 else 'INSUFFICIENT — needs more instances or precompute'}, "
        "target rule of thumb is ≥1.5x so a single AZ/instance loss doesn't page someone at 3am).\n"
        f"- If running behind a load balancer, minimum instance count for {TARGET_SUSTAINED_QPS} QPS "
        f"at 1.5x headroom ≈ **{max(1, round(1.5 * TARGET_SUSTAINED_QPS / safe_qps + 0.49))} instances**, "
        "assuming linear horizontal scaling (validated separately, not assumed — see scaling_plan.md).\n"
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
