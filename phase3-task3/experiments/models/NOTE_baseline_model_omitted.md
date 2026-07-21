# Baseline model artifact omitted from this package

`ranker_baseline_v1.joblib` (272 MB, 400 unbounded-depth trees) was excluded
from this ZIP purely for file-size practicality — its size is itself part of
the evidence for why optimization was needed (see README "TL;DR result":
285.1 MB -> 0.22 MB, a 99.9% reduction).

To regenerate it exactly (same random seed, deterministic):

```bash
pip install -r requirements.txt
python3 -m data.generate_data          # regenerates data/interaction_logs*.csv
python3 -m src.train_baseline          # regenerates experiments/models/ranker_baseline_v1.joblib
```

Everything else in this package (metrics, latency profiles, the optimized
model, reports) was produced by a full run that included this model — it is
not missing from the results, only from the packaged file tree.
