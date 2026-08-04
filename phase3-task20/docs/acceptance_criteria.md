# Acceptance Criteria — AcmeFinServ_Pilot
(Stage A: "the customer's definition of a good match, agreed in advance.")

This is the **proposal to review with the tenant's hiring manager**
before the real pilot — not a decision made on their behalf.

## What "good" means for this tenant
1. **Ranking quality:** top-10 candidates per requisition should contain
   more real hires than the recruiter's current manual skill-keyword
   sort (baseline). Decided by: `hire_capture@10` on held-out jobs.
2. **Fairness:** selection rate ratio (min/max across protected groups)
   must stay ≥ 0.8 (4/5ths rule) AND equal-opportunity rates (rate at
   which truly-hired candidates were surfaced in top 10) must not
   diverge by more than 2 percentage points between groups.
3. **Latency:** p99 < 200ms to rank a full requisition's candidate pool
   (current pilot measured well under this — see experiments/latency_report.json).
4. **Explainability:** every ranked candidate must have a one-line,
   plain-English reason a recruiter can read without ML background.
5. **Degradation:** if the model is unavailable, the tenant must still
   get a ranked list (baseline fallback), never an error page.

## Baseline being beaten
Skill-keyword overlap ranking (`skill_overlap`), representing what the
tenant's recruiters currently do manually.

## What is explicitly NOT yet agreed (open items for the pilot kickoff)
- Whether hire_capture@10 or nDCG@10 is the primary go/no-go metric —
  this run found they disagree (see experiments/metrics_offline.json),
  so the tenant must pick which one they trust before the real pilot.
- Minimum acceptable equal-opportunity gap threshold (2pp is a starting
  proposal, not yet signed off).
- Whether a human recruiter must review every shortlist before it's
  sent to a hiring manager (human-in-the-loop vs fully automated —
  see Section 8 of the study guide, "Alternative approaches").
