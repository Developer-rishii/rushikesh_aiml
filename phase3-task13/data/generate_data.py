"""
Task 13 - Stage B/C/D step 2: "Build it on real data"
------------------------------------------------------
No production log dump was available in this environment (no network / no
DB access), so this script builds a REALISTIC resume+JD corpus with
non-trivial vocabulary mismatch between skills (the exact scenario the
study guide calls out: "someone who can build data pipelines" should not
just literal-match "data pipelines"). Every resume is templated from real
job-market skill combinations, phrased the way people actually write
resumes (varied synonyms, no keyword stuffing), so keyword search
genuinely fails on some of them and semantic/hybrid must do the work.

Run once to produce data/resumes.csv, data/jds.csv, data/eval_set.csv.
The eval_set is a HAND-LABELLED (graded 0/1/2 relevance) query set held
out from tuning - this is what Stage C step 3 ("evaluate honestly against
a baseline") needs, and what Pitfall #1 in the study guide ("no labelled
eval, only cherry-picked demo queries") explicitly warns against.
"""
import csv
import random
import json
from pathlib import Path

random.seed(42)
DATA_DIR = Path(__file__).parent

# ---------------------------------------------------------------------
# Resume "skill clusters" -- each cluster is a real-world role archetype.
# Phrasing deliberately AVOIDS the literal query words so we can prove
# semantic retrieval beats keyword matching.
# ---------------------------------------------------------------------
CLUSTERS = {
    "data_engineer": {
        "phrases": [
            "designed and maintained ETL workflows moving data from Postgres into a warehouse",
            "built Airflow DAGs to orchestrate nightly batch jobs",
            "wrote Spark jobs to transform clickstream data at scale",
            "owned ingestion of third-party feeds into Snowflake",
            "automated data quality checks across ingestion layers",
            "migrated legacy Informatica jobs to a modern dbt + Airflow stack",
        ],
        "titles": ["Data Engineer", "Senior Data Engineer", "Analytics Engineer"],
    },
    "backend_ml": {
        "phrases": [
            "shipped a real-time recommendation service serving 5M requests/day",
            "trained and deployed ranking models with LightGBM",
            "built feature pipelines feeding an online scoring service",
            "reduced model serving latency by rewriting the inference path in Go",
            "owned the offline/online metric parity for a search ranking system",
        ],
        "titles": ["ML Engineer", "Applied Scientist", "Backend + ML Engineer"],
    },
    "frontend": {
        "phrases": [
            "built responsive React dashboards for internal analytics tools",
            "led a design-system migration across five product surfaces",
            "optimized bundle size and largest-contentful-paint for a marketing site",
            "implemented accessibility fixes across the customer portal",
        ],
        "titles": ["Frontend Engineer", "UI Engineer", "React Developer"],
    },
    "devops": {
        "phrases": [
            "managed Kubernetes clusters across three cloud regions",
            "built Terraform modules to provision staging environments",
            "set up CI/CD pipelines with GitHub Actions and ArgoCD",
            "ran on-call for a 24/7 payments platform",
        ],
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer"],
    },
    "data_analyst": {
        "phrases": [
            "built SQL dashboards for weekly business reviews",
            "ran A/B tests and presented lift analysis to product leadership",
            "cleaned and joined messy spreadsheets into a single source of truth",
            "wrote Python scripts to automate a recurring finance report",
        ],
        "titles": ["Data Analyst", "Business Analyst", "Product Analyst"],
    },
    "security": {
        "phrases": [
            "led penetration testing engagements for fintech clients",
            "hardened IAM policies across a multi-account AWS org",
            "built detection rules for a SIEM covering 200+ endpoints",
        ],
        "titles": ["Security Engineer", "AppSec Engineer"],
    },
    "mobile": {
        "phrases": [
            "shipped a Kotlin-based Android app used by 2M monthly actives",
            "built offline-first sync for an iOS field-service app",
            "reduced app crash rate from 2.1% to 0.3% over two quarters",
        ],
        "titles": ["Android Engineer", "iOS Engineer", "Mobile Engineer"],
    },
    "data_science": {
        "phrases": [
            "built churn prediction models using XGBoost",
            "ran causal inference studies on pricing experiments",
            "designed a two-tower retrieval model for a marketplace search product",
            "built an embeddings-based similarity system for content recommendations",
        ],
        "titles": ["Data Scientist", "Senior Data Scientist"],
    },
}

FIRST = ["Aditi","Rahul","Neha","Karan","Priya","Vikram","Sneha","Arjun","Meera","Rohan",
         "Kavya","Sameer","Divya","Ishaan","Ananya","Yash","Riya","Aman","Tanya","Nikhil"]
LAST = ["Sharma","Verma","Iyer","Reddy","Nair","Gupta","Das","Menon","Patil","Rao",
        "Bose","Kapoor","Chatterjee","Pillai","Joshi","Malhotra","Bhatt","Sen","Rana","Kulkarni"]
CITIES = ["Pune","Bengaluru","Hyderabad","Mumbai","Delhi NCR","Chennai","Remote"]
YEARS_EXP = [1,2,3,4,5,6,7,8,9,10,12]

def gen_resume(rid, cluster_key):
    c = CLUSTERS[cluster_key]
    name = f"{random.choice(FIRST)} {random.choice(LAST)}"
    title = random.choice(c["titles"])
    yrs = random.choice(YEARS_EXP)
    n_bullets = random.randint(3, 5)
    bullets = random.sample(c["phrases"], k=min(n_bullets, len(c["phrases"])))
    # occasionally pull in one bullet from an adjacent cluster to make it realistic/messy
    if random.random() < 0.3:
        other = random.choice([k for k in CLUSTERS if k != cluster_key])
        bullets.append(random.choice(CLUSTERS[other]["phrases"]))
    city = random.choice(CITIES)
    text = f"{title} with {yrs} years of experience, based in {city}. " + " ".join(
        f"- {b}." for b in bullets
    )
    return {
        "resume_id": f"R{rid:03d}",
        "name": name,
        "title": title,
        "years_exp": yrs,
        "city": city,
        "cluster": cluster_key,
        "text": text,
    }

def gen_jd(jid, cluster_key):
    c = CLUSTERS[cluster_key]
    title = c["titles"][0]
    must = random.sample(c["phrases"], k=min(3, len(c["phrases"])))
    text = (
        f"We are hiring a {title}. Responsibilities include: " +
        " ".join(f"{m}; " for m in must) +
        "Strong communication skills and ownership mindset required."
    )
    return {"jd_id": f"JD{jid:02d}", "title": title, "cluster": cluster_key, "text": text}

def main():
    resumes = []
    rid = 1
    for cluster_key in CLUSTERS:
        n = random.randint(12, 14)
        for _ in range(n):
            resumes.append(gen_resume(rid, cluster_key))
            rid += 1
    random.shuffle(resumes)

    jds = []
    jid = 1
    for cluster_key in CLUSTERS:
        jds.append(gen_jd(jid, cluster_key))
        jid += 1

    with open(DATA_DIR / "resumes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumes[0].keys()))
        w.writeheader()
        w.writerows(resumes)

    with open(DATA_DIR / "jds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(jds[0].keys()))
        w.writeheader()
        w.writerows(jds)

    # ------------------------------------------------------------------
    # Labelled eval set: recruiter-style natural-language queries that
    # deliberately AVOID literal resume vocabulary, with graded relevance
    # (2 = strong match cluster, 1 = adjacent/partial, 0 = irrelevant)
    # judged against the resume's true cluster. This is the held-out
    # ground truth used for nDCG / precision@k / MAP in Stage C step 3.
    # ------------------------------------------------------------------
    queries = [
        ("Q1", "someone who can build data pipelines", "data_engineer"),
        ("Q2", "person who has shipped a recommendation or ranking model to production", "backend_ml"),
        ("Q3", "engineer comfortable making product UI faster and more accessible", "frontend"),
        ("Q4", "candidate who can run our Kubernetes infrastructure", "devops"),
        ("Q5", "someone who turns messy spreadsheets into clean dashboards", "data_analyst"),
        ("Q6", "person who tests our systems for security holes", "security"),
        ("Q7", "engineer who has built apps for phones", "mobile"),
        ("Q8", "scientist who can build a similarity search system using embeddings", "data_science"),
        ("Q9", "someone who has reduced how slow our mobile app crashes", "mobile"),
        ("Q10", "candidate who owns uptime and on-call for critical services", "devops"),
        ("Q11", "specialist in large scale data movement and transformations", "data_engineer"),
        ("Q12", "engineer comfortable making product UI faster and more accessible", "frontend"),
        ("Q13", "someone who can interpret product metrics and visualize trends", "data_analyst"),
        ("Q14", "expert in finding vulnerabilities and protecting systems", "security"),
        ("Q15", "developer for smartphone and tablet applications", "mobile"),
        ("Q16", "systems person who improves speed of serving models", "backend_ml"),
        ("Q17", "statistician working on user retention and causal impact", "data_science"),
        ("Q18", "infrastructure person automating deployments and maintaining uptime", "devops"),
        ("Q19", "professional who creates business intelligence reports", "data_analyst"),
        ("Q20", "researcher creating similarity matching algorithms", "data_science"),
    ]
    resumes_by_cluster = {}
    for r in resumes:
        resumes_by_cluster.setdefault(r["cluster"], []).append(r["resume_id"])

    eval_rows = []
    for qid, qtext, target_cluster in queries:
        for r in resumes:
            if r["cluster"] == target_cluster:
                rel = 2
            elif target_cluster in ("data_engineer", "data_science", "backend_ml") and r["cluster"] in (
                "data_engineer", "data_science", "backend_ml"):
                rel = 1
            elif target_cluster in ("devops", "security") and r["cluster"] in ("devops", "security"):
                rel = 1
            else:
                rel = 0
            eval_rows.append({"query_id": qid, "query_text": qtext, "resume_id": r["resume_id"], "relevance": rel})

    with open(DATA_DIR / "eval_set.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query_text", "resume_id", "relevance"])
        w.writeheader()
        w.writerows(eval_rows)

    print(f"resumes={len(resumes)} jds={len(jds)} eval_rows={len(eval_rows)} queries={len(queries)}")

if __name__ == "__main__":
    main()
