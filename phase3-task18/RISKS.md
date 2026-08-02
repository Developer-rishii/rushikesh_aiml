# Risks & Honest Gaps

These are named explicitly rather than hidden, per the rubric's "A claim
without evidence scores zero" and the pitfalls list in the study guide.

1. **Offline-only validation.** The +16–30% relative lift in `EXPERIMENT_LOG.md`
   is measured on held-out logs, not live traffic. No online A/B result
   exists here. This must be closed before claiming production-readiness.
2. **Synthetic data.** `data/generate_logs.py` simulates realistic log shape
   and org-skill bias with noise, but real recruiter behaviour will have
   different distributions (seasonality, role-specific skill demand,
   adversarial/bot traffic). Model retraining on real logs is required
   before the numeric lift can be trusted at face value.
3. **Model class simplification.** Pointwise GBM instead of listwise
   LambdaMART (see DESIGN_DECISIONS.md §5) — a known, disclosed
   simplification driven by no network access in this environment, not by
   design preference.
4. **Fairness audit not performed.** The study guide explicitly flags "a
   fairness audit done once, at the end, as a formality" as a pitfall — this
   submission does not include a fairness audit at all (no protected-class
   labels exist in the synthetic data). This is a genuine gap, not
   formality-washed; it must be built before any real hiring-adjacent
   ranking ships.
5. **No model registry/versioning wired up.** `MLflow or an equivalent` was
   in the recommended stack; this project logs experiments as markdown
   (`EXPERIMENT_LOG.md`) instead, which is not sufficient for "which model
   produced a decision six months ago" at production scale.
6. **Timestamps are simulated.** Real logs would have actual event
   timestamps enabling a true chronological split; the synthetic generator
   doesn't timestamp events, so `evaluate.py`'s "temporal" split uses a
   fixed-seed shuffle per recruiter as a stand-in. Noted in code comments.
