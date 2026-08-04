# Remediation List — Before the Real Pilot

Tenant: AcmeFinServ_Pilot

## [HIGH] Model underperforms the skill_overlap baseline on offline MAP@10 (-0.0047) and Precision@10 (-0.11) even though it wins on the real-outcome hire-capture proxy (0.3 more hires captured in top 10).

**Action:** Do not ship on offline nDCG/MAP alone. Run an online A/B test on a small traffic slice before the full pilot; the hire-capture proxy suggests real value the ranking metrics understate, but this must be confirmed online, not assumed.

## [HIGH] Demographic parity passes the 4/5ths rule (ratio=0.8993), but equal-opportunity rates diverge sharply between groups (A=0.019, B=0.0465): among candidates who actually get hired, group B is surfaced in the top 10 far more often than group A.

**Action:** Passing demographic parity is not sufficient. Investigate why truly-qualified group-A candidates rank lower before the real pilot -- check for skill-vocabulary or experience-distribution confounds; add equal-opportunity as a hard gate, not just parity.

## [MEDIUM] LambdaMART/listwise learning-to-rank was rejected for this dry-run due to no LightGBM/XGBoost available in the build sandbox; a pointwise GradientBoostingRegressor proxy was used instead.

**Action:** Before the real pilot, retrain with a listwise objective (LambdaMART) in an environment with package access, since pointwise regression optimizes per-row accuracy, not result ORDER, which the guide identifies as what actually drives outcomes.

## [MEDIUM] Domain shift: 60% of tenant-facing job titles use internal vocabulary (e.g. 'Quant Modeling Specialist' vs standard 'Data Scientist') that differs from generic training data title strings.

**Action:** Build a tenant title-vocabulary mapping table (see TENANT_TITLE_VOCAB in data/generate_data.py as the pattern) and validate it with the tenant's HR team before the real pilot; do not rely on the model to learn tenant vocabulary from too few examples.

## [LOW] p99 latency to rank a full 2,000-candidate pool for one requisition: 36.82 ms.

**Action:** Within budget for a pilot (<100ms p99). Before scaling past this tenant, re-benchmark with the real candidate pool size and add caching for repeated job/candidate feature lookups.

## [MEDIUM] No model versioning/registry existed before this run.

**Action:** Adopted: every trained model is now saved with a version tag and a model card (models/model_card.md) recording training data window, features, and metrics, so a decision can be traced back to the exact model version months later (guide pitfall: 'cannot say which model produced a decision six months ago').

## [MEDIUM] Acceptance criteria for 'a good match' were assumed, not agreed with a real enterprise stakeholder.

**Action:** Before the real pilot: get the tenant's hiring manager to sign off explicitly on what 'good' means (which offline/online metric, what threshold) -- see docs/acceptance_criteria.md draft for the starting proposal to review with them.
