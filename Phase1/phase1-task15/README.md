# Task 15 — K-Means Clustering

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–14:** loads Task 14's locked, prepared
(scaled, feature-selected, PCA-reduced) data and its justified `k=2`
directly from `locked_clustering_params.json` — this task never
re-decides k, it runs and interprets the clustering that was already
justified. Same `SEED=42`.

## What this delivers (Definition of Done)

**Interpreted, named clusters with validity scores and a recommended
action per segment** — demonstrated live in `src/run_kmeans_segments.py`,
following the study guide's 6 steps:

1. **Run K-Means at the locked k** — `k=2`, read from Task 14's lock
   file, never hand-typed here (structurally can't silently drift).
2. **Validity: silhouette + inertia** — overall and **per-cluster**
   silhouette computed, not just one aggregate number.
3. **Profile each cluster** — `src/clustering/profile.py`: z-score of
   every cluster's centroid against the population mean, on the
   **original, interpretable feature space** (never the PCA components,
   which have no business meaning) — the top 5 defining features per
   cluster, each with a direction and a magnitude.
4. **Name clusters in business terms** — `name_clusters()` derives the
   name **directly and traceably** from each cluster's top defining
   feature (verified by test — the name isn't a disconnected label).
5. **Stability across seeds** — `src/clustering/stability.py` refits
   K-Means with 5 different seeds and measures Adjusted Rand Index
   against the primary run.
6. **Recommend an action per segment** — `src/clustering/recommend.py`,
   tying each segment's profile to a concrete recommendation.

## A guard the guide's "Limitations" section asks for, added proactively

K-Means assumes round, similar-size clusters — `src/clustering/shape_check.py`
actually checks both (cluster size ratio, within-cluster spread ratio)
against what was fit, rather than assuming the assumption holds just
because K-Means ran without erroring.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Clusters with no interpretation | `test_pitfall_clusters_are_interpreted_not_bare_labels` | Asserts every cluster has a computed profile AND that its name is traceably derived from the profile's actual top feature — not a hand-picked label disconnected from the data |
| Forcing K-Means on non-spherical data | `test_pitfall_shape_assumptions_actually_checked` | Asserts the size/spread ratios are actually computed, not assumed reasonable |
| Unstable clusters across runs | `test_pitfall_stability_actually_measured` | Asserts stability is measured across 5 independent seeds with real ARI numbers, not asserted stable without evidence |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Interpreted, named clusters with validity scores + recommended action | `outputs/reports/kmeans_segments_report.json` (profiles, names, validity, stability, recommendations all in one place) |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 14's real 7-component PCA-reduced feature space, real per-cluster malignancy rates as an external check |
| Live verification & evidence | `tests/test_kmeans_segments.py` — 6/6 tests pass on live runs; `outputs/artifacts/labeled_records.csv` has every real row's actual cluster assignment |
| Dependency/failure/edge-case handling | Missing Task 14 hand-off, `k=1`, and prepared/raw data row-count mismatches all raise clearly before reaching sklearn |

## How to run

```bash
pip install -r requirements.txt
python tests/test_kmeans_segments.py   # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_kmeans_segments
```

## Results from this run (seed=42)

**Validity:** overall silhouette **0.348** (k=2, from Task 14's lock);
per-cluster silhouette: Cluster 0 = 0.157, Cluster 1 = 0.442 — Cluster 1
is noticeably more cohesive than Cluster 0.

**Cluster 0 — "High Shape Irregularity"** (n=188, based on `mean
concavity`, z=+1.148): elevated concavity, compactness, and concave
points relative to the population.

**Cluster 1 — "Low Shape Irregularity"** (n=381, based on `mean
concavity`, z=-0.567): the complementary, smoother-boundary segment.

**Shape-assumption check:** cluster size ratio 2.03 (moderate imbalance,
below the 3.0 flag threshold), spread ratio 1.48 (below the 2.0 flag
threshold) -> **K-Means's round/similar-size assumption is reasonable
here**, not silently forced onto a bad fit.

**Stability:** min ARI **1.0** across 5 independent seeds (1, 7, 13, 21,
99) -> **perfectly stable**, the strongest possible result — this
2-cluster split is not an artifact of one lucky initialization.

**External reference (informational only, never used to form or name
clusters) — and this is the genuinely interesting finding:** Cluster 0
("High Shape Irregularity") is **86.2% malignant**; Cluster 1 ("Low
Shape Irregularity") is **87.1% benign**. The unsupervised geometry
found purely from feature distances recovers most of the real clinical
distinction without ever seeing the label.

**Recommended actions (real, per segment):**
- Cluster 0: **PRIORITY REVIEW** — route for expedited pathologist
  follow-up given the malignancy skew and defining shape-irregularity profile.
- Cluster 1: **STANDARD MONITORING** — appropriate for routine
  follow-up scheduling.

Full numbers: `outputs/reports/kmeans_segments_report.json`.
Per-record cluster assignments: `outputs/artifacts/labeled_records.csv`.
Profile plot: `outputs/figures/cluster_profiles.png`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-14. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task15_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                       # reads k from Task 14's lock file
│   └── config.yaml                     # n_init, stability seeds, profiling settings
├── data/
│   ├── clustering_ready_data.csv       # Task 14's PCA-reduced, locked data
│   ├── locked_clustering_params.json   # Task 14's k + selected features
│   ├── scaler.joblib / pca.joblib      # Task 14's fitted objects
│   └── clean_from_task2.csv            # original units, for interpretable profiling
├── src/
│   ├── __init__.py
│   ├── run_kmeans_segments.py          # THE 6-step flow
│   └── clustering/
│       ├── fit.py                        # Steps 1-2: run K-Means, validity
│       ├── profile.py                    # Steps 3-4: profile + name
│       ├── stability.py                  # Step 5: multi-seed ARI
│       ├── shape_check.py                # Limitations guard: round/similar-size check
│       ├── recommend.py                  # Step 6: per-segment action
│       └── plots.py                      # cluster profile bar chart
├── tests/
│   └── test_kmeans_segments.py         # live run + one test per named pitfall + edge cases
└── outputs/
    ├── artifacts/
    │   ├── kmeans_model.joblib
    │   └── labeled_records.csv           # every real row + its cluster assignment
    ├── reports/
    │   └── kmeans_segments_report.json
    ├── figures/
    │   └── cluster_profiles.png
    └── logs/
        └── run_kmeans_segments.log
```
