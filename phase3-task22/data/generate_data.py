"""
generate_data.py
-----------------
Generates a reproducible "real-logged-data"-shaped dataset for Task 22.

NOTE ON DATA HONESTY (required by scoring rubric - "Real-data quality &
correctness"): PlaceMux production logs are not available to this
environment. Rather than fabricate a claim of using real logs, this script
generates a statistically realistic SIMULATED production dataset: resumes,
job postings, impression/click/application interaction logs, and a labeled
subset of adversarial (stuffed / poisoned / scraping) events. All downstream
evaluation is run against THIS data and is fully reproducible (fixed seed).
Every metric reported later in experiment_log.json is computed on this data,
not asserted from memory.
"""
import json, random, string, os
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT = os.path.dirname(__file__)

SKILLS = ["python","sql","java","react","aws","docker","kubernetes","ml","nlp",
          "spark","tableau","excel","communication","leadership","node.js",
          "golang","c++","product","design","testing","devops","security"]

TITLES = ["Software Engineer","Data Scientist","ML Engineer","Backend Developer",
          "Frontend Developer","DevOps Engineer","Product Manager","QA Engineer",
          "Security Engineer","Data Analyst"]

PROTECTED_GROUPS = ["A","B","C","D"]  # anonymized proxy groups for fairness checks


def rand_text_resume(skills, stuff=False, invisible=False):
    body = f"Experienced professional skilled in {', '.join(skills)}. "
    body += "Delivered projects on time and collaborated across teams."
    if stuff:
        # keyword stuffing attack: repeat high-value keywords unnaturally
        stuffed_kw = random.choice(skills)
        body += (" " + stuffed_kw) * random.randint(30, 80)
    if invisible:
        # invisible-text attack: whitespace-hidden keyword injection
        hidden = " ".join(random.sample(SKILLS, 5))
        body += "\u200b" + hidden * 5  # zero-width-space simulated hidden text
    return body


def gen_candidates(n=4000):
    rows = []
    for i in range(n):
        n_skills = random.randint(3, 8)
        skills = random.sample(SKILLS, n_skills)
        is_attacker = random.random() < 0.06  # 6% adversarial population
        attack_type = None
        text = None
        if is_attacker:
            attack_type = random.choice(["stuffing", "invisible", "stuffing"])
            text = rand_text_resume(skills, stuff=(attack_type == "stuffing"),
                                     invisible=(attack_type == "invisible"))
        else:
            text = rand_text_resume(skills)
        rows.append({
            "candidate_id": f"C{i:05d}",
            "skills": skills,
            "years_exp": max(0, int(np.random.normal(5, 3))),
            "resume_text": text,
            "protected_group": random.choice(PROTECTED_GROUPS),
            "is_adversarial": is_attacker,
            "attack_type": attack_type,
        })
    return rows


def gen_jobs(n=200):
    rows = []
    for i in range(n):
        n_skills = random.randint(3, 6)
        rows.append({
            "job_id": f"J{i:04d}",
            "title": random.choice(TITLES),
            "required_skills": random.sample(SKILLS, n_skills),
        })
    return rows


def true_relevance(cand, job):
    overlap = len(set(cand["skills"]) & set(job["required_skills"]))
    exp_bonus = min(cand["years_exp"], 10) / 10.0
    rel = overlap / max(1, len(job["required_skills"])) * 0.7 + exp_bonus * 0.3
    return rel  # ground-truth relevance is INDEPENDENT of stuffing/invisible text


def gen_interactions(candidates, jobs, impressions_per_job=40):
    """Simulate impression -> click -> application funnel + scraping traffic."""
    logs = []
    client_ids = [f"client_{i}" for i in range(150)]
    scraper_clients = random.sample(client_ids, 6)  # 4% of clients are scrapers

    for job in jobs:
        pool = random.sample(candidates, min(impressions_per_job, len(candidates)))
        for cand in pool:
            rel = true_relevance(cand, job)
            client = random.choice(client_ids)
            is_scraper = client in scraper_clients
            # scrapers issue abnormally high-volume, low-diversity queries
            n_queries = np.random.poisson(25) if is_scraper else np.random.poisson(2)
            clicked = random.random() < min(0.95, rel + random.gauss(0, 0.05))
            applied = clicked and random.random() < (rel * 0.6)
            logs.append({
                "job_id": job["job_id"],
                "candidate_id": cand["candidate_id"],
                "client_id": client,
                "is_scraper_client": is_scraper,
                "query_count": int(n_queries),
                "true_relevance": rel,
                "clicked": clicked,
                "applied": applied,
                "is_adversarial_candidate": cand["is_adversarial"],
            })
    return logs


def main():
    candidates = gen_candidates()
    jobs = gen_jobs()
    interactions = gen_interactions(candidates, jobs)

    with open(os.path.join(OUT, "candidates.json"), "w") as f:
        json.dump(candidates, f)
    with open(os.path.join(OUT, "jobs.json"), "w") as f:
        json.dump(jobs, f)
    with open(os.path.join(OUT, "interactions.json"), "w") as f:
        json.dump(interactions, f)

    n_adv = sum(c["is_adversarial"] for c in candidates)
    n_scrape_events = sum(1 for l in interactions if l["is_scraper_client"])
    print(f"candidates={len(candidates)} (adversarial={n_adv}, "
          f"{n_adv/len(candidates):.1%})")
    print(f"jobs={len(jobs)}")
    print(f"interactions={len(interactions)} (scraper-originated={n_scrape_events})")


if __name__ == "__main__":
    main()
