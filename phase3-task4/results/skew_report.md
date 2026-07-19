# Train/Serve Skew Report

Training rows: 24,000 · Served (sampled) rows logged live during load test + failure demo: 20,906

| feature | train_mean | served_mean | train_std | z | flagged |
|---|---|---|---|---|---|
| exp_years | 4.031 | 5.999 | 2.848 | 0.69 | YES |
| skill_match | 0.501 | 0.410 | 0.223 | 0.41 | no |
| education_score | 0.501 | 0.500 | 0.288 | 0.01 | no |
| location_match | 0.500 | 0.334 | 0.500 | 0.33 | no |
| past_response_rate | 0.286 | 0.300 | 0.161 | 0.09 | no |
| job_seniority | 1.482 | 2.000 | 1.104 | 0.47 | no |
| job_urgency_score | 0.497 | 0.600 | 0.290 | 0.35 | no |
| job_num_applicants_so_far | 99.425 | 39.986 | 58.884 | 1.01 | YES |

**Skew detected** — investigate feature pipeline before trusting model scores.
