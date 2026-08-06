# Threat Model — PlaceMux Matching/Ranking ML System

## 1. What "good" means & the bar (Stage B.1)
Good = every attacker class below has a named, testable defence with a
detection metric, AND the system degrades safely (never silently wrong) if
a defence component itself fails. The bar: **a candidate cannot game their
way to the top; an attacker cannot steal or poison the model** — and both
claims must be demonstrated live (Section 5), not asserted.

## 2. Assets
- Ranking/matching model weights & training pipeline
- Candidate resumes, PII, protected-group attributes
- Historical interaction logs (impressions/clicks/applications)
- Ranked results served at query time

## 3. STRIDE threat enumeration

| # | Threat class | STRIDE | Attack | Impact | Defence (built in this repo) |
|---|---|---|---|---|---|
| T1 | Keyword stuffing | Tampering | Candidate repeats high-value keywords / invisible unicode text to inflate relevance score | Unfair ranking, hiring-quality loss | `ranking_defense/stuffing_detector.py` — rule signals (keyword-repetition rate, char-entropy, zero-width chars) + supervised classifier; detected resumes are down-weighted, not silently ranked |
| T2 | Ranking manipulation via feature gaming | Tampering | Candidate reverse-engineers ranker features (e.g. spamming skill list) | Same as T1 | Robust featurization: skill-match features are **capped/deduplicated** (max 1 credit per unique skill) so repetition yields zero marginal gain — see `ranker.py::robust_features` |
| T3 | Model extraction | Information Disclosure | Competitor/bot issues high-volume systematic queries to reconstruct decision boundary | IP theft, competitive loss | `extraction_poison_detection/extraction_detector.py` — per-client query-rate + query-diversity (entropy) monitoring, rate limiting & anomaly flag |
| T4 | Training-data poisoning | Tampering | Attacker feeds mislabeled/synthetic interactions into the log so future retraining shifts rankings in their favor | Slow, hard-to-detect model corruption | `extraction_poison_detection/poison_detector.py` — Isolation-Forest outlier detection on (feature, label) consistency before any data enters retraining |
| T5 | Fairness / disparate impact | (cross-cutting) | Any of T1–T4 could disproportionately help/hurt a protected group | Legal/DPDP exposure | Fairness slice reported every eval run (`evaluate.py`), not a one-time formality (Pitfall #4 explicitly avoided) |
| T6 | Model unavailability | Denial of Service | Ranking service crashes/times out | Broken candidate experience | `integration/failure_injection.py` — verified fallback to a deterministic recency-based ranking, never a raw crash or unranked list |
| T7 | Model versioning / accountability gap | (Repudiation) | Cannot say which model produced a decision N months ago | Compliance failure (Pitfall #5) | Every run in `run_pipeline.py` writes a versioned entry to `experiment_log.json` with model hash + timestamp |

## 4. What good defences must NOT do
- Must not simply **block** all high-signal candidates (false positives hurt
  real candidates) — see Section 8 "alternative approaches" for the
  block-vs-downrank tradeoff we chose (downrank + flag for review).
- Must not rely on a single offline metric — every defence is checked
  offline AND against an online-effect proxy (clicks/applications), because
  offline wins that never validate online are an explicit named pitfall.

## 5. Live verification plan (executed in `integration/attack_simulation.py`)
1. Inject a keyword-stuffed resume at query time → assert it is detected
   AND its effective rank drops vs. an honest higher-relevance resume.
2. Simulate a scraping client issuing 25x normal query volume with low
   query diversity → assert it is flagged and rate-limited.
3. Inject 3% poisoned interaction rows into a retraining batch → assert
   the poison detector isolates them before they reach the trainer.
4. Kill the ranking model mid-request → assert the system falls back to
   the recency ranker instead of erroring or serving unranked results.

## 6. Rejected alternatives (Stage A.3 / Section 8 of study guide)
- **Rule-based-only stuffing detection** was rejected as the sole defence:
  attackers adapt to fixed rules quickly. We use rules as fast, explainable
  first-pass signals BUT combine them with a trained classifier that can
  generalize to unseen stuffing patterns (hybrid, not either/or).
- **Hard blocking of flagged accounts** was rejected as the default action:
  false positives (a candidate who legitimately lists many relevant skills)
  would be locked out with no recourse. Default action is **silent
  down-ranking + human-review queue**; hard block is reserved for
  extraction/scraping clients (T3) where false-positive cost is much lower
  (a client, not a job-seeking human).
