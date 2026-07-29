# Bugfix Log — post-delivery evaluation

## Bug found: hardcoded absolute paths
Every script wrote to / read from `/home/claude/placemux_task14/...` literally.
This worked in the sandbox that built it (that exact folder still existed)
but **fails on any fresh machine or renamed folder** — exactly what happened
when this project was re-extracted as `phase3-task14/`. Verified failure:

```
FileNotFoundError: [Errno 2] No such file or directory:
'/home/claude/placemux_task14/data/interactions_log.csv'
```

This is precisely the kind of "looks done but isn't" failure the study
guide's pitfalls section warns about (§12) — silent, only surfaces when
someone other than the original author runs it.

## Fix
Added `src/paths.py`: resolves `ROOT` relative to the script's own file
location (`os.path.dirname(...__file__...)`), so the project works no
matter where it's unzipped or what it's renamed to. All 5 affected files
(`generate_data.py`, `train_model.py`, `mitigation.py`, `api.py`,
`failure_demo.py`) now import from it instead of hardcoding paths.

## Re-verification after fix
Ran the full pipeline from a **completely clean state** (deleted all
generated files first, in the renamed `phase3-task14/` folder, with the
original build folder removed from disk entirely) to rule out any
leftover-file false positive:

| Step | Result |
|---|---|
| `data/generate_data.py` | ✅ writes to local `data/interactions_log.csv` |
| `src/train_model.py` | ✅ AUC 0.7218, EOD -0.0329 (identical to original numbers) |
| `src/mitigation.py` | ✅ EOD -0.0194, same 41% improvement |
| `src/failure_demo.py` | ✅ 503 DEFERRED_TO_HUMAN_REVIEW on injected failure, recovers after |
| `tests/test_fairness.py` (7 tests) | ✅ 7/7 passed |
| `src/api.py` as a real standalone server (not test client) | ✅ `curl localhost:5000/explain` returns correct exact attribution |

All numbers reproduced byte-for-byte identical to the original report —
confirms the bug was purely a path-portability issue, not a correctness
issue in the actual audit/mitigation/explainability logic.
