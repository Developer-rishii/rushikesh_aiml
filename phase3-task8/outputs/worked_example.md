# Worked Example — Candidate C105970

**As-of date:** 2025-11-01
**Model risk score:** 0.928
**Actual outcome (next 21 days):** churned (no activity)

**Why the model flagged this candidate (plain English):**
- activity in the last 90 days is lower than typical (z=-1.2)
- activity in the last 30 days is lower than typical (z=-1.2)
- account age is higher than typical (z=1.1)

**If the model is unavailable:** If the model service is down, this candidate would instead be scored by the 14-day-inactivity rule using days_since_last_event only -- degraded but never silent (see failure_simulation.py).