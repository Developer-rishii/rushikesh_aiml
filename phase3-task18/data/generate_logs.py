"""
Generates synthetic but realistic recruiter/org interaction logs for PlaceMux.

Why synthetic here: no production log access exists outside Altrodav's systems.
This generator simulates the SAME shape as real logs (impressions -> clicks ->
shortlists -> applications, keyed by recruiter_id + org_id + candidate_id) so the
scoped-signal, lifecycle and isolation logic below can be built and evaluated
against something with realistic noise, class imbalance and org boundaries.
Swap this module for a real log loader in production; nothing downstream cares.
"""
import json
import random
from pathlib import Path

random.seed(42)

N_ORGS = 12
RECRUITERS_PER_ORG = (2, 5)
N_CANDIDATES = 800
IMPRESSIONS_PER_RECRUITER = (150, 400)

SKILLS = ["python", "sql", "react", "node", "aws", "ml", "java", "go",
          "product", "sales", "design", "devops", "data-eng", "qa"]

ORG_SKILL_BIAS = {f"org_{i}": random.sample(SKILLS, 4) for i in range(N_ORGS)}


def make_candidate(cid):
    return {
        "candidate_id": cid,
        "skills": random.sample(SKILLS, random.randint(2, 5)),
        "years_exp": round(random.uniform(0, 12), 1),
    }


def gen_recruiter_events(recruiter_id, org_id, candidates, preferred_skills):
    events = []
    n = random.randint(*IMPRESSIONS_PER_RECRUITER)
    for _ in range(n):
        cand = random.choice(candidates)
        overlap = len(set(cand["skills"]) & set(preferred_skills))
        # click probability driven by org-specific skill affinity (the real signal
        # a recruiter-scoped model should learn) plus noise
        click_p = min(0.85, 0.05 + 0.22 * overlap)
        clicked = random.random() < click_p
        shortlisted = clicked and random.random() < (0.15 + 0.1 * overlap)
        applied = shortlisted and random.random() < 0.4
        events.append({
            "recruiter_id": recruiter_id,
            "org_id": org_id,
            "candidate_id": cand["candidate_id"],
            "candidate_skills": cand["skills"],
            "candidate_years_exp": cand["years_exp"],
            "clicked": int(clicked),
            "shortlisted": int(shortlisted),
            "applied": int(applied),
        })
    return events


def generate(out_path):
    candidates = [make_candidate(f"cand_{i}") for i in range(N_CANDIDATES)]
    all_events = []
    recruiter_org_map = {}
    rid_counter = 0
    for oi in range(N_ORGS):
        org_id = f"org_{oi}"
        n_rec = random.randint(*RECRUITERS_PER_ORG)
        for _ in range(n_rec):
            rid = f"rec_{rid_counter}"
            rid_counter += 1
            recruiter_org_map[rid] = org_id
            events = gen_recruiter_events(rid, org_id, candidates, ORG_SKILL_BIAS[org_id])
            all_events.extend(events)

    Path(out_path).write_text(json.dumps({
        "events": all_events,
        "recruiter_org_map": recruiter_org_map,
        "org_skill_bias": ORG_SKILL_BIAS,
    }, indent=0))
    return len(all_events), len(recruiter_org_map)


if __name__ == "__main__":
    out = Path(__file__).parent / "interaction_logs.json"
    n_events, n_rec = generate(out)
    print(f"generated {n_events} events across {n_rec} recruiters -> {out}")
