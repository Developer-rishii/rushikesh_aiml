"""
Generate a realistic synthetic dataset of ~500 raw parsed skill strings.
Mirrors what "Parsing v0" would output from resumes and JDs.

Categories of raw terms generated:
  - exact_match: exact display name
  - abbreviation: known synonym / abbreviation
  - typo: misspelled version of a known skill
  - multi_word: skill embedded in a phrase (e.g. "Python (3 yrs)")
  - unmappable: parsing noise / artifacts

Outputs:
  data/synthetic_raw_terms.json  — full dataset with labels
  data/test_set.json             — held-out test split (30%)
  data/train_set.json            — remaining 70% (for threshold tuning only)
"""

import json
import random
import os
import string

random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def introduce_typo(word: str) -> str:
    """Introduce a single-character typo into a word."""
    if len(word) < 3:
        return word
    idx = random.randint(1, len(word) - 2)
    mutation = random.choice(["swap", "delete", "insert", "replace"])
    chars = list(word)
    if mutation == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif mutation == "delete":
        chars.pop(idx)
    elif mutation == "insert":
        chars.insert(idx, random.choice(string.ascii_lowercase))
    elif mutation == "replace":
        chars[idx] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def wrap_with_experience(skill_str: str) -> str:
    """Wrap a skill string with experience context like resumes have."""
    templates = [
        "{skill} (3 yrs)",
        "{skill} - 5 years",
        "{skill} 2+ years experience",
        "Sr. {skill} Developer",
        "{skill} Engineer",
        "Lead {skill} Dev",
        "{skill}/Full-Stack",
        "Junior {skill} Developer",
        "{skill} (intermediate)",
        "{skill}, expert level",
    ]
    return random.choice(templates).format(skill=skill_str)


UNMAPPABLE_NOISE = [
    "•", "●", "skills:", "SKILLS", ":", "–", "—", "·",
    "proficient in", "experienced with", "familiar with",
    "strong knowledge of", "exposure to", "worked on",
    "xkq7z", "a3f#$%", "........", "N/A", "n/a",
    "TBD", "see resume", "various tools", "etc.",
    "hard worker", "team player", "self-motivated",
    "References available", "Curriculum Vitae", "Page 2",
    "zzz123", "qwerty", "asdfgh", "Lorem ipsum",
    "Jan 2020 - Dec 2022", "123-456-7890", "email@example.com",
    "skills & abilities", "TECHNICAL SKILLS", "KEY COMPETENCIES",
    "2019", "GPA 3.5", "B.Tech", "M.S. in CS",
    "University of XYZ", "Certifications:", "Awards:",
    "µ§∆", "OCR_FRAGMENT_ERROR", "\\x00\\x01",
    "...", "---", "***", "###", "|||",
    "responsibilities included", "duties:", "summary:",
    "Professional Experience", "Education", "Contact Information",
    "", "   ", "\t", "\n",
    "asdkjhf aslkjdf", "xyzzy plugh", "fnord",
    "12345", "!@#$%", "()", "[]", "{}",
]


def generate_dataset(ontology_path: str) -> list[dict]:
    """Generate ~500 labelled raw terms from the ontology."""
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    records: list[dict] = []

    # ---- Exact matches (~20%) ----
    for skill in random.sample(ontology, min(70, len(ontology))):
        records.append({
            "raw_term": skill["display_name"],
            "expected_canonical_id": skill["canonical_id"],
            "segment": "exact_match",
        })
    # Add some with different casing
    for skill in random.sample(ontology, min(40, len(ontology))):
        cased = random.choice([
            skill["display_name"].lower(),
            skill["display_name"].upper(),
            skill["display_name"].title(),
        ])
        records.append({
            "raw_term": cased,
            "expected_canonical_id": skill["canonical_id"],
            "segment": "exact_match",
        })

    # ---- Abbreviation / synonym matches (~25%) ----
    for skill in ontology:
        for syn in skill.get("synonyms", []):
            if syn.lower() != skill["display_name"].lower():
                records.append({
                    "raw_term": random.choice([syn, syn.upper(), syn.title()]),
                    "expected_canonical_id": skill["canonical_id"],
                    "segment": "abbreviation",
                })
    # Keep ~125 abbreviation records
    abbrev_records = [r for r in records if r["segment"] == "abbreviation"]
    if len(abbrev_records) > 130:
        to_remove = random.sample(abbrev_records, len(abbrev_records) - 130)
        for r in to_remove:
            records.remove(r)

    # ---- Typo variants (~15%) ----
    for skill in random.sample(ontology, min(75, len(ontology))):
        base = random.choice([skill["display_name"]] + skill.get("synonyms", []))
        typo_version = introduce_typo(base.lower())
        if typo_version.lower() != base.lower():  # only if actually different
            records.append({
                "raw_term": typo_version,
                "expected_canonical_id": skill["canonical_id"],
                "segment": "typo",
            })

    # ---- Multi-word / experience-wrapped (~15%) ----
    for skill in random.sample(ontology, min(75, len(ontology))):
        base = random.choice([skill["display_name"]] + skill.get("synonyms", []))
        records.append({
            "raw_term": wrap_with_experience(base),
            "expected_canonical_id": skill["canonical_id"],
            "segment": "multi_word",
        })

    # ---- Unmappable noise (~10-15%) ----
    num_unmappable = max(60, int(len(records) * 0.13))
    noise_terms = random.choices(UNMAPPABLE_NOISE, k=num_unmappable)
    for term in noise_terms:
        records.append({
            "raw_term": term,
            "expected_canonical_id": "unmapped",
            "segment": "unmappable",
        })

    random.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Parsing v0 Loader — drop-in replacement for synthetic data
# ---------------------------------------------------------------------------

def load_parsing_v0_output(filepath: str) -> list[dict]:
    """
    Load real Parsing v0 output.  Expected schema (one JSON array):
        [
          {"raw_term": "ReactJS",  "source": "resume", "student_id": "stu_001"},
          {"raw_term": "ML",       "source": "jd",     "student_id": null},
          ...
        ]

    DEPENDENCY MITIGATION PLAN
    --------------------------
    This loader is the ONE line that swaps synthetic → real data.
    If Parsing v0 is late or its schema changes:
      1. We keep serving the synthetic dataset (last-known-good).
      2. This function logs a warning and falls back to synthetic.
      3. When Parsing v0 delivers, update the schema mapping below
         and re-run evaluation to confirm no metric regression.
      4. Escalation: notify Parsing v0 team lead within 24 hours of
         missed delivery date — do NOT wait silently.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Schema validation
        if not isinstance(raw_data, list):
            raise ValueError("Expected a JSON array at top level")

        records = []
        for item in raw_data:
            if "raw_term" not in item:
                logger.warning(f"Skipping record without 'raw_term': {item}")
                continue
            records.append({
                "raw_term": str(item["raw_term"]),
                "source": item.get("source", "unknown"),
                "student_id": item.get("student_id"),
            })

        logger.info(f"Loaded {len(records)} terms from Parsing v0 output at {filepath}")
        return records

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to load Parsing v0 output: {e}. Falling back to synthetic data.")
        return None  # caller should fall back to synthetic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    ontology_path = os.path.join("data", "ontology.json")

    records = generate_dataset(ontology_path)
    print(f"Total records generated: {len(records)}")

    # Segment breakdown
    from collections import Counter
    seg_counts = Counter(r["segment"] for r in records)
    for seg, count in sorted(seg_counts.items()):
        print(f"  {seg}: {count} ({count/len(records)*100:.1f}%)")

    # Save full dataset
    with open(os.path.join("data", "synthetic_raw_terms.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Split into train (70%) and test (30%) — stratified by segment
    from collections import defaultdict
    by_segment = defaultdict(list)
    for r in records:
        by_segment[r["segment"]].append(r)

    train, test = [], []
    for seg, items in by_segment.items():
        random.shuffle(items)
        split_idx = int(len(items) * 0.7)
        train.extend(items[:split_idx])
        test.extend(items[split_idx:])

    random.shuffle(train)
    random.shuffle(test)

    with open(os.path.join("data", "train_set.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join("data", "test_set.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)

    print(f"\nTrain set: {len(train)} records")
    print(f"Test set:  {len(test)} records (held out — never used for tuning)")
