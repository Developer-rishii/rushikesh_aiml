# PlaceMux Live Model Monitoring - Demo Walkthrough

## One real example, end-to-end

- match_id: `P002200`
- match probability: **0.0662**
- predicted label: **no match**
- decision threshold: 0.48

**Why:** Scored 7% probability -> classified as an unlikely match. Main driver: weak verified skill overlap with the JD (0.01); secondary factor: interview evaluation score (0.20, against match).

| feature | value | contribution | direction |
|---|---|---|---|
| skill_overlap_score | 0.012 | -1.7003 | against match |
| interview_eval_score | 0.202 | -0.5315 | against match |
| experience_gap | 2.652 | -0.3374 | against match |

## Batch-by-batch monitoring summary

| batch | n_total | n_labeled | n_pending | precision | recall | FPR | drift |
|---|---|---|---|---|---|---|---|
| 0 | 200 | 98 | 102 | 0.26 | 0.8125 | 0.4512 | warning |
| 1 | 200 | 108 | 92 | 0.4138 | 0.7742 | 0.4416 | stable |
| 2 | 200 | 106 | 94 | 0.3966 | 0.697 | 0.4795 | stable |
| 3 | 200 | 119 | 81 | 0.3529 | 0.7742 | 0.5 | stable |
| 4 | 200 | 104 | 96 | 0.4237 | 0.7576 | 0.4789 | warning |
| 5 | 200 | 111 | 89 | 0.2679 | 0.6 | 0.4767 | stable |
| 6 | 200 | 107 | 93 | 0.2364 | 0.7222 | 0.4719 | stable |
| 7 | 200 | 114 | 86 | 0.25 | 0.3333 | 0.1875 | critical |
| 8 | 200 | 105 | 95 | 0.4286 | 0.3333 | 0.0417 | critical |
| 9 | 200 | 122 | 78 | 0.0 | 0.0 | 0.0 | critical |
| 10 | 200 | 110 | 90 | 0.0 | 0.0 | 0.0 | critical |
| 11 | 200 | 109 | 91 | 0.0 | 0.0 | 0.0 | critical |

Total alerts raised across the run: **17**
