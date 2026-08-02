# Next Steps (study guide §14 "Go deeper")

1. **Listwise LambdaMART.** Swap `GradientBoostingRegressor` in
   `src/ranking_model.py` for LightGBM's `LGBMRanker` (objective="lambdarank")
   once network/package access exists. Evaluation harness in `evaluate.py`
   is already group-aware and needs no other change.
2. **Two-tower retrieval + ANN index.** Current model is a reranker over all
   candidates; add a two-tower embedding retrieval stage (candidate tower +
   recruiter/org-scoped query tower) with FAISS/pgvector for the top-K
   candidate generation step before ranking, for marketplace-scale serving.
3. **Counterfactual / off-policy evaluation.** Add IPS or doubly-robust
   estimators to translate the offline nDCG/MAP lift into an expected online
   effect estimate before requesting production A/B traffic.
4. **Fairness audit.** Build a protected-attribute-aware (or proxy-aware)
   parity check comparing scoped vs baseline ranking outcomes, run
   continuously (not once at the end) — closes the gap in `RISKS.md` §4.
5. **Model registry.** Wire MLflow (or equivalent) so every model version in
   `EXPERIMENT_LOG.md` has a queryable artifact + lineage, satisfying "which
   model produced a decision six months ago."
6. **Real timestamps.** Once real logs are available, replace the per-
   recruiter shuffle stand-in in `evaluate.py::temporal_split_per_recruiter`
   with a true chronological cutoff.
