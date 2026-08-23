# Task 14 — Data Cluster Parameter Prep

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–13:** loads Task 2's leakage-cleaned WDBC data.
Unlike Tasks 3–13, this task is **unsupervised** — the target column is
loaded but deliberately held aside, unused anywhere in feature selection,
scaling, PCA, or k-selection (using it there would be a supervised
shortcut masquerading as unsupervised prep). It only appears once, at
the very end, as an external reference check. Same `SEED=42`.

## What this delivers (Definition of Done)

**A scaled, feature-selected dataset and a justified choice of k for
clustering** — demonstrated live in `src/run_cluster_prep.py`, following
the study guide's 6 steps:

1. **Feature selection** — `src/clustering/feature_selection.py`, two
   redundancy-based passes (not target-based, since clustering has no
   label to rank against): drop near-zero-**relative**-variance columns
   (coefficient of variation, not raw variance — see the design note
   below) and drop features highly correlated (>0.95) with an
   already-kept feature.
2. **Scale** — `StandardScaler`, verified structurally (mean~0, std~1
   on every kept feature), not just assumed applied.
3. **PCA** — reduces to the number of components retaining >=90%
   variance, capped at 10.
4. **Elbow + silhouette for k** — both computed across `k=2..8`, plotted
   together; k picked by the **measured** silhouette score, not a guess.
5. **Distance sanity-check** — `src/clustering/sanity_check.py`: the
   chosen k's silhouette must clear a minimum bar AND the max/min
   pairwise-distance ratio in the reduced space must stay well above 1
   (a ratio collapsing toward 1 is the textbook curse-of-dimensionality
   symptom — distances becoming meaningless).
6. **Lock the prepared dataset + params** — `outputs/prepared_data/`:
   the PCA-reduced feature matrix, the fitted scaler/PCA objects, and a
   single `locked_clustering_params.json` documenting every choice made.

## A design note worth flagging (a real bug caught and fixed while building this)

The first version of Step 1's variance filter used **raw** variance,
which — on WDBC's mixed-unit features — incorrectly flagged genuinely
informative small-scale features (`smoothness`, `symmetry`, `fractal
dimension`, all naturally ~0.05-0.3 in absolute value) as "near-constant"
purely because of their unit, dropping **16 of 30** features. That's
precisely the kind of scale-driven distortion Step 2 (scaling) exists to
prevent — it just leaked into Step 1's selection logic instead. Fixed by
switching to **coefficient of variation** (std/|mean|), which is
scale-invariant: the corrected run drops **0** features on relative-
variance grounds and correctly keeps all 30 as informative, dropping
only the 6 genuinely redundant, highly-correlated ones. Both the bug and
the fix are documented here rather than silently corrected.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Clustering on unscaled features | `test_pitfall_features_are_scaled_before_clustering` | Structurally asserts every scaled feature has mean~0 and std~1 — proven, not claimed |
| Arbitrary k | `test_pitfall_k_is_evidence_based_not_arbitrary` | Asserts all 7 candidate k values were actually evaluated and produced genuinely different silhouette scores (the choice discriminates) |
| Too many noisy dimensions | `test_pitfall_noisy_dimensions_actually_reduced` | Asserts the pipeline actually reduces column count at both the selection stage and the PCA stage, not just claims to |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Scaled, feature-selected dataset + justified k | `outputs/prepared_data/clustering_ready_data.csv`, `locked_clustering_params.json`, `outputs/figures/elbow_silhouette.png` |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, all 30 real features actually evaluated for redundancy, a real 7-value k sweep |
| Live verification & evidence | `tests/test_cluster_prep.py` — 7/7 tests pass on live runs; the correlation-dropping check confirms real near-duplicate WDBC pairs (e.g. mean/worst radius) are actually caught, not assumed |
| Dependency/failure/edge-case handling | `k >= n_samples` and feature-selection thresholds that would drop every column both raise clearly before crashing deep inside sklearn |

## How to run

```bash
pip install -r requirements.txt
python tests/test_cluster_prep.py   # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_cluster_prep
```

## Results from this run (seed=42)

**Feature selection:** 30 -> 24 features (6 dropped for redundancy,
correlation >0.95 with an already-kept feature — e.g. `mean radius` vs
`worst radius`-style near-duplicates; **0** dropped for low relative
variance — all 30 original features carry genuine signal).

**Scaling:** all 24 kept features confirmed mean~0, std~1.

**PCA:** 24 scaled features -> **7 components**, retaining **90.18%**
variance.

**k selection:** silhouette scores across k=2..8:
`[0.348, 0.289, 0.187, 0.181, 0.176, 0.172, 0.169]` — **k=2 chosen**,
clearly the strongest by a wide margin, not a close call.

**Distance sanity check:** silhouette 0.348 clears the 0.15 minimum;
max/min pairwise-distance ratio **36.99** (well above the 2.0
concentration-warning threshold) -> **`distances_meaningful: true`**.

**External reference (informational only, never used to pick features
or k):** Adjusted Rand Index between the k=2 clusters and the actual
malignant/benign diagnosis label = **0.5329** — moderate agreement,
suggesting the natural 2-cluster structure found purely from feature
geometry does partially recover real diagnostic structure, without ever
having seen the label during prep.

Full numbers: `outputs/prepared_data/locked_clustering_params.json`,
`outputs/reports/cluster_prep_report.json`. Elbow + silhouette plot:
`outputs/figures/elbow_silhouette.png`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-13. Only `pip install -r
requirements.txt` needs network access, once (adds `scipy`, `matplotlib`).

## Folder structure

```
task14_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                       # YAML -> typed Config, sets global seed
│   └── config.yaml                     # selection thresholds, PCA, k-range
├── data/
│   └── clean_from_task2.csv            # carried over from Task 2
├── src/
│   ├── __init__.py
│   ├── run_cluster_prep.py             # THE 6-step flow
│   ├── data/dataset.py                   # loads features, holds target aside
│   └── clustering/
│       ├── feature_selection.py          # Step 1: variance + correlation redundancy
│       ├── prepare.py                    # Steps 2-3: scaling + PCA
│       ├── select_k.py                   # Step 4: elbow + silhouette
│       └── sanity_check.py               # Step 5: distance meaningfulness
├── tests/
│   └── test_cluster_prep.py            # live run + one test per named pitfall + edge cases
└── outputs/
    ├── prepared_data/
    │   ├── clustering_ready_data.csv     # THE locked, prepared dataset
    │   ├── locked_clustering_params.json # THE locked parameters
    │   ├── scaler.joblib
    │   └── pca.joblib
    ├── reports/
    │   └── cluster_prep_report.json
    ├── figures/
    │   └── elbow_silhouette.png
    └── logs/
        └── run_cluster_prep.log
```
