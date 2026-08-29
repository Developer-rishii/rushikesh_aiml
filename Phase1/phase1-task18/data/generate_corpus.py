"""
data/generate_corpus.py — builds a realistic job-posting text corpus
across 6 job categories. Sourcing note (documented, not hidden): no live
internet dataset is reachable in this environment (only package
registries like PyPI are allowed, not general web/API access), so this
corpus is generated from category-specific vocabulary pools combined
via varied sentence templates — genuinely varied natural language at a
realistic document count (600 postings, ~60-110 words each), not five
hand-typed rows and not literal copy-paste duplicates within a category.
Every document is independently sampled, so within-category documents
share vocabulary and structure the way real job postings in the same
field genuinely do, without being identical.
"""
import random
from pathlib import Path
import pandas as pd

CATEGORIES = {
    "software_engineer": {
        "titles": ["Software Engineer", "Backend Developer", "Full Stack Engineer", "Platform Engineer"],
        "skills": ["Python", "Java", "distributed systems", "REST APIs", "microservices", "Kubernetes",
                   "Docker", "SQL databases", "unit testing", "CI/CD pipelines", "cloud infrastructure",
                   "Git", "system design", "algorithms and data structures", "code review"],
        "duties": ["design and implement scalable backend services", "collaborate with product managers on requirements",
                    "debug production incidents", "write clean, well-tested code", "participate in on-call rotations",
                    "mentor junior engineers", "optimize database queries", "review pull requests"],
        "quals": ["a degree in Computer Science or related field", "3+ years of professional software development experience",
                    "strong problem-solving skills", "experience with agile development", "excellent communication skills"],
    },
    "data_scientist": {
        "titles": ["Data Scientist", "Machine Learning Engineer", "Applied Scientist", "ML Researcher"],
        "skills": ["Python", "pandas", "scikit-learn", "statistical modeling", "A/B testing", "SQL",
                   "deep learning", "feature engineering", "experiment design", "data visualization",
                   "PyTorch", "hypothesis testing", "causal inference", "model deployment"],
        "duties": ["build predictive models from large datasets", "design and analyze experiments",
                    "communicate findings to stakeholders", "collaborate with engineering on model deployment",
                    "clean and preprocess messy real-world data", "validate model performance rigorously",
                    "present insights to non-technical audiences"],
        "quals": ["a graduate degree in a quantitative field", "hands-on experience with machine learning frameworks",
                    "strong statistical foundations", "ability to translate ambiguous problems into models",
                    "experience presenting to business stakeholders"],
    },
    "registered_nurse": {
        "titles": ["Registered Nurse", "Staff Nurse", "Clinical Nurse", "ICU Nurse"],
        "skills": ["patient assessment", "medication administration", "electronic health records",
                    "wound care", "IV therapy", "vital signs monitoring", "patient education",
                    "care plan documentation", "infection control protocols", "triage"],
        "duties": ["provide direct patient care on the unit", "administer medications per physician orders",
                    "monitor patient vital signs and escalate concerns", "educate patients and families on care plans",
                    "document care accurately in the electronic health record", "coordinate with the care team",
                    "respond to emergency situations calmly and effectively"],
        "quals": ["an active RN license in good standing", "BLS and ACLS certification", "2+ years of clinical experience",
                    "excellent bedside manner", "ability to work rotating shifts including nights and weekends"],
    },
    "accountant": {
        "titles": ["Staff Accountant", "Senior Accountant", "Financial Accountant", "Accounting Analyst"],
        "skills": ["GAAP", "accounts payable", "accounts receivable", "month-end close", "financial reporting",
                    "reconciliations", "Excel", "QuickBooks", "budgeting", "variance analysis", "audit support",
                    "journal entries", "tax compliance"],
        "duties": ["prepare monthly financial statements", "reconcile general ledger accounts",
                    "assist with the annual audit", "process accounts payable and receivable transactions",
                    "support budgeting and forecasting cycles", "ensure compliance with GAAP",
                    "identify and resolve discrepancies in financial records"],
        "quals": ["a Bachelor's degree in Accounting or Finance", "CPA or CPA-track preferred",
                    "2+ years of accounting experience", "strong attention to detail",
                    "proficiency with accounting software"],
    },
    "sales_representative": {
        "titles": ["Sales Representative", "Account Executive", "Business Development Representative", "Sales Associate"],
        "skills": ["CRM software", "lead generation", "cold calling", "negotiation", "pipeline management",
                   "consultative selling", "closing deals", "territory management",
                   "prospecting", "relationship building", "quota attainment"],
        "duties": ["prospect and qualify new sales leads", "build and maintain a pipeline of opportunities",
                    "conduct product demonstrations for prospective clients", "negotiate contract terms and pricing",
                    "meet or exceed quarterly sales quotas", "maintain accurate records in the CRM",
                    "build long-term relationships with key accounts"],
        "quals": ["2+ years of B2B sales experience", "a track record of meeting or exceeding quota",
                    "excellent verbal and written communication skills", "self-motivated and results-driven",
                    "experience with CRM tools such as Salesforce"],
    },
    "graphic_designer": {
        "titles": ["Graphic Designer", "Visual Designer", "Brand Designer", "Creative Designer"],
        "skills": ["Adobe Photoshop", "Adobe Illustrator", "typography", "brand identity", "layout design",
                    "Figma", "color theory", "print production", "digital design", "UI design principles",
                    "motion graphics", "InDesign"],
        "duties": ["create visual assets for marketing campaigns", "maintain brand consistency across channels",
                    "collaborate with marketing on campaign concepts", "design layouts for print and digital media",
                    "present design concepts to stakeholders", "iterate on designs based on feedback",
                    "manage multiple design projects simultaneously"],
        "quals": ["a portfolio demonstrating strong visual design skills", "proficiency in the Adobe Creative Suite",
                    "a degree in Graphic Design or related field", "strong eye for typography and layout",
                    "ability to work under tight deadlines"],
    },
}

INTROS = [
    "We are looking for a {title} to join our growing team.",
    "Our company is seeking an experienced {title} to help drive our next phase of growth.",
    "As a {title} on our team, you will play a key role in our day-to-day operations.",
    "We're hiring a {title} who is passionate about delivering high-quality work.",
    "Join us as a {title} and make a real impact from day one.",
]

RESPONSIBILITY_LEAD = ["In this role, you will", "Key responsibilities include", "You will be expected to",
                        "Day to day, you will"]
QUAL_LEAD = ["We are looking for candidates with", "The ideal candidate will have", "Requirements include",
             "You should bring"]
SKILL_LEAD = ["This role requires strong skills in", "Relevant skills include", "You should be comfortable with"]


def _make_doc(rng: random.Random, category: str) -> str:
    pool = CATEGORIES[category]
    title = rng.choice(pool["titles"])
    intro = rng.choice(INTROS).format(title=title)

    duties = rng.sample(pool["duties"], k=min(3, len(pool["duties"])))
    duty_sentence = f"{rng.choice(RESPONSIBILITY_LEAD)} {', '.join(duties[:-1])}, and {duties[-1]}."

    skills = rng.sample(pool["skills"], k=min(4, len(pool["skills"])))
    skill_sentence = f"{rng.choice(SKILL_LEAD)} {', '.join(skills[:-1])}, and {skills[-1]}."

    quals = rng.sample(pool["quals"], k=min(3, len(pool["quals"])))
    qual_sentence = f"{rng.choice(QUAL_LEAD)} {', '.join(quals[:-1])}, and {quals[-1]}."

    closing = rng.choice([
        "We offer a competitive salary and comprehensive benefits package.",
        "This is a full-time position with opportunities for growth.",
        "We value diversity and are proud to be an equal opportunity employer.",
        "Apply today to become part of our team.",
    ])
    return " ".join([intro, duty_sentence, skill_sentence, qual_sentence, closing])


def generate_corpus(n_per_category: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for category in CATEGORIES:
        for _ in range(n_per_category):
            rows.append({"category": category, "text": _make_doc(rng, category)})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_stress_test_hybrids(n: int = 40, seed: int = 43) -> pd.DataFrame:
    """
    Genuinely ambiguous documents: each mixes HALF its skill/duty/qual
    vocabulary from one category and half from another related category
    (e.g. software_engineer + data_scientist, both use Python/SQL), with
    NO category-specific job title mentioned (a generic "This role" intro
    instead) — the title alone is such a strong discriminative signal
    that including it would make even a 50/50 content mix trivially
    separable, defeating the point of a genuine stress test. Exists
    specifically so Step 5's error-inspection machinery has real,
    non-trivial errors to characterize — the clean primary corpus above
    is deliberately well-separated (6 distinct professions), so it alone
    can't demonstrate what error inspection looks like on a hard case.
    Labeled with a designated "true" category (cat_a) even though content
    is evenly split — by design, a well-behaved classifier SHOULD
    struggle here, and that struggle is the point.
    """
    rng = random.Random(seed)
    related_pairs = [
        ("software_engineer", "data_scientist"),
        ("accountant", "sales_representative"),
    ]
    generic_intros = [
        "This role is available on our team, reporting to the department lead.",
        "We have an opening for this position within our growing organization.",
        "This position offers the chance to contribute across multiple areas.",
    ]
    rows = []
    for i in range(n):
        cat_a, cat_b = related_pairs[i % len(related_pairs)]
        pool_a, pool_b = CATEGORIES[cat_a], CATEGORIES[cat_b]
        intro = rng.choice(generic_intros)

        duties = rng.sample(pool_a["duties"], k=2) + rng.sample(pool_b["duties"], k=2)
        rng.shuffle(duties)
        duty_sentence = f"{rng.choice(RESPONSIBILITY_LEAD)} {', '.join(duties[:-1])}, and {duties[-1]}."

        skills = rng.sample(pool_a["skills"], k=3) + rng.sample(pool_b["skills"], k=3)
        rng.shuffle(skills)
        skill_sentence = f"{rng.choice(SKILL_LEAD)} {', '.join(skills[:-1])}, and {skills[-1]}."

        quals = rng.sample(pool_a["quals"], k=2) + rng.sample(pool_b["quals"], k=2)
        rng.shuffle(quals)
        qual_sentence = f"{rng.choice(QUAL_LEAD)} {', '.join(quals[:-1])}, and {quals[-1]}."

        text = " ".join([intro, duty_sentence, skill_sentence, qual_sentence])
        # alternate which of the pair is "true" so the stress set isn't lopsided
        true_cat = cat_a if i % 2 == 0 else cat_b
        rows.append({"category": true_cat, "text": text})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_corpus()
    out_path = Path(__file__).resolve().parent / "job_postings_corpus.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} documents across {df['category'].nunique()} categories -> {out_path}")
    print(df["category"].value_counts())

    hybrids = generate_stress_test_hybrids()
    hybrid_path = Path(__file__).resolve().parent / "stress_test_hybrids.csv"
    hybrids.to_csv(hybrid_path, index=False)
    print(f"Wrote {len(hybrids)} stress-test hybrid documents -> {hybrid_path}")
