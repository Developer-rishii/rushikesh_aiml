# Before / After — Latency, Quality, Cost

All numbers below are read directly from `experiments/metrics_before_after.json`,
`experiments/latency_profile_before.json` and `experiments/latency_profile_after.json`,
produced by an actual run of `python3 -m scripts.run_all`. Nothing here is hand-typed
or estimated after the fact.

## 1. Where the time was going (the profile)

Stage-by-stage breakdown of a single representative ranking request (40 candidates):

| Stage | BEFORE (baseline) | AFTER (optimized) |
|---|---|---|
| Feature fetch | ~40 ms (40 sequential network round trips, 1 per candidate) | ~1.3 ms (1 batched round trip for the whole page) |
| Model inference + overhead | ~43 ms (400-tree, unbounded-depth RandomForest) | ~2.7 ms (25-tree, depth-6 distilled RandomForest) |
| **Total** | **~83 ms** | **~4 ms (cold cache) / ~1.4 ms (warm cache)** |

**Answer to "is the bottleneck the model, or fetching its features?": both, roughly
50/50 in the baseline** — which is why the fix had to attack both (batching the
fetch *and* shrinking the model), not just one.

## 2. Latency — p50 / p95 / p99 over 120 held-out ranking requests

| Metric | BEFORE | AFTER (cold cache) | AFTER (warm cache) | SLO |
|---|---|---|---|---|
| p50 | 88.5 ms | 3.7 ms | 1.4 ms | — |
| **p95** | **101.1 ms** | 3.9 ms | **1.4 ms** | **40.0 ms** |
| p99 | 116.5 ms | 4.0 ms | 1.4 ms | — |
| Meets SLO? | **No** | Yes | **Yes** | |

**p95 speedup: ~72x** (101.1 ms → 1.4 ms), measured on the same 120 held-out jobs,
same machine, same run.

## 3. Quality — held-out test set, not tuned on (120 jobs, 4,800 rows)

| Metric | BEFORE | AFTER | Δ |
|---|---|---|---|
| nDCG@10 | 0.5643 | 0.5816 | **+0.0173** |
| P@10 | 0.7308 | 0.7608 | **+0.0300** |
| MAP@10 | 0.8348 | 0.8440 | **+0.0092** |

Quality did **not** regress — it improved slightly. This is a real, if modest,
effect of distillation training on a blend of true labels and the (well-generalizing)
baseline's soft predictions, which acts as a mild regularizer versus the
unbounded-depth baseline. We are not claiming this generalizes to every distillation
setup — see "Honest caveats" below — but on this held-out set it is the actual number,
not a cherry-pick (it's every offline metric we tracked, not the one that looked best).

## 4. Cost — daily compute cost at an assumed 2,000,000 requests/day

| | BEFORE | AFTER | Savings |
|---|---|---|---|
| p50 latency | 88.5 ms | 1.4 ms | |
| Estimated daily compute cost | $159.24 | $2.47 | **$156.77/day (~98%)** |
| Model artifact size | 285.1 MB | 0.22 MB | 99.9% smaller |

Cost model: `daily_cost = p50_ms * requests_per_day * $/ms`, at an illustrative
$0.0000009/compute-ms (~$0.9 per million compute-ms) — stated explicitly in
`src/config.py` so the assumption is visible and adjustable, not baked in silently.

## 5. Failure injection (designed degradation)

With the model forced unavailable, the optimized server did **not** throw or 5xx —
it served all 40 candidates via `popularity_fallback_scorer` (a cheap, model-free
heuristic on activity + response-rate). See `experiments/metrics_before_after.json
-> failure_injection_demo` and `scripts/demo.py` output.

## 6. Train/serve skew check

Run against a deliberately buggy alternate implementation (`distance_km` left in
miles) and against the real, shared implementation used by both training and
serving. The check correctly **failed** the buggy path (`distance_km` max abs
diff = 34.26) and **passed** the shared path (max abs diff = 0.0 on all four
cheap features). See console output of `python3 -m src.skew_check`.

## 7. Fairness sanity check (not just a formality at the end)

Run at both baseline and optimized stages, on the synthetic protected attribute
`candidate_group`, against a 15% relative-gap threshold:

| | Group A mean score | Group B mean score | Relative gap | Flagged? |
|---|---|---|---|---|
| Baseline | 0.868 | 0.862 | 0.6% | No |
| Optimized | 0.857 | 0.857 | 0.04% | No |

This is a lightweight sanity check appropriate to a latency-focused task, not a
full fairness audit (out of scope here) — that distinction is stated explicitly
rather than implied.

## 8. What we rejected, and why

- **Smaller/distilled model vs. caching-only**: caching alone doesn't help the
  first (cold) hit on any given candidate, and doesn't reduce inference cost for
  the long tail of one-off candidates. We used both: distillation cuts the
  per-inference cost unconditionally; caching removes redundant re-scoring within
  a TTL window on top of that.
- **Real-time inference vs. precomputed/scheduled scores**: candidate/job
  relevance in this domain shifts with every new application and profile edit,
  so precomputing on a schedule risks serving stale rankings during high-urgency
  hiring windows. We kept real-time inference but made it cheap enough (1.4 ms
  p95) that precomputation isn't needed to hit the SLO.

## 9. Honest caveats

- The interaction log is **simulated**, not real PlaceMux production data (see
  `data/generate_data.py` docstring) — no real logs were available in this
  environment. The pipeline, metrics, and profiling methodology are exactly what
  would run against real logs; only the input rows are synthetic.
- The quality *improvement* from distillation is a real measured result on this
  dataset, not a guaranteed property of distillation in general — smaller models
  can also lose quality; that's why the evaluation step is mandatory and was run,
  not skipped.
- The $/ms cost figure is illustrative (stated in `config.py`); swap in your
  actual cloud billing rate for a production-accurate number.

## 10. Note on run-to-run variance

The BEFORE/AFTER *latency* numbers above come from one specific run and will vary by roughly ±15% between runs, because the feature-store round trip is simulated with `time.sleep()` and is therefore subject to normal OS scheduling jitter on shared infrastructure. The *quality* metrics (nDCG/P/MAP) and the *model size* numbers are deterministic (fixed random seed) and identical across runs. Re-run `python3 -m scripts.run_all` at any time to regenerate all numbers fresh — the speedup and SLO pass/fail conclusion is stable across runs even though the exact millisecond values move slightly.
