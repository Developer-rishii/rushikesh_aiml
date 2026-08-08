# Offline Evaluation Report

Evaluated on a held-out split (25% of jobs, unseen during training), grouped by job_id, nDCG@10.

- Held-out rows: 4900
- Held-out distinct jobs: 30
- **Model nDCG@10:** 0.2893
- **Heuristic baseline nDCG@10:** 0.3131
- **Lift over baseline:** -7.6%

Interpretation: this offline gap is what we EXPECT to translate into an online application-rate lift. Per the study guide, offline wins are not shipped as truth -- they must be validated online (A/B) before being trusted; this report only clears the offline bar and documents the gap to be checked against.
