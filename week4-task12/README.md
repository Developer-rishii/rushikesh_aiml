# Task 12 — Parsing v0

AI/ML Engineer deliverable for Week 4, Phase 2 — E-Sign Integration & Tamper-Evidence — PlaceMux, Altrodav Technologies.

The team-wide theme this week is e-sign integration and tamper-evidence verification, but that is a **different role's deliverable**. This slice is strictly **Parsing v0**: a ground-up resume/JD parser that extracts structured skills, normalized against a skills ontology, ready to hand off as structured profiles and jobs to the next team. The self-check questions in the original study guide regarding e-signing offers were written for the backend e-sign deliverable, not this one. This mismatch is documented and e-sign features are explicitly excluded from this repository.

## Upstream Dependency: Skills Ontology

This parser normalizes text against a strictly required upstream dependency: the skills ontology.
- **File**: `data/skills_ontology.csv`
- **Size**: 66 canonical skills
- **Structure**: `canonical_name`, `category` (14 languages, 14 concepts, 11 frameworks, etc.), and `aliases` (pipe-separated synonyms/abbreviations).

The pipeline validates this input explicitly at load time rather than assuming it is well-formed. It checks for required columns, duplicate canonical names, and empty alias lists. This validation is proven by the `test_ontology_validation_malformed` test in `tests/test_edge_cases.py`, which ensures the pipeline fails loudly if the ontology is malformed. This parser is new ground-up parsing work, not a continuation of any prior model.

## Extraction Evaluation Metrics

**Verdict: ✅ Full pipeline beats the unfiltered extractor on precision.** The ML confidence filter effectively removes false positives, increasing overall precision from 0.8203 to 0.9987, at a negligible cost to recall (0.9987 down to 0.9974).

Metrics below are computed on a full held-out sample of 199 synthetic resumes and 80 JDs, deliberately injected with hard edge cases.

### Overall Performance

| Method | Precision | Recall | FPR | TP | FP | FN |
|--------|-----------|--------|-----|----|----|-----|
| Baseline | 0.9941 | 0.8814 | 0.0059 | 669 | 4 | 90 |
| Rules-Only (3a) | 0.8203 | 0.9987 | 0.1797 | 758 | 166 | 1 |
| Full Pipeline (3a+3b) | 0.9987 | 0.9974 | 0.0013 | 757 | 1 | 2 |

### Hard-Case Breakdown

| Edge Case Type | Method | Precision | Recall | FPR | Details |
|---|---|---|---|---|---|
| **Alias-only** (no canonical names) | Rules-Only (3a) | 0.8571 | 0.8571 | 0.1429 | FPs: ['Terraform'], Misses: ['TensorFlow'] |
| | Full Pipeline | 0.8333 | 0.7143 | 0.1667 | FPs: ['Terraform'], Misses: ['TensorFlow', 'R'] |
| **Negation** ("no experience with X") | Rules-Only (3a) | 0.5000 | 1.0000 | 0.5000 | FPs: ['Leadership'], Misses: none |
| | Full Pipeline | 1.0000 | 1.0000 | 0.0000 | FPs: none, Misses: none |
| **Substring trap** ("R" in "Director") | Rules-Only (3a) | 0.3333 | 1.0000 | 0.6667 | FPs: ['Data Engineering', 'R'], Misses: none |
| | Full Pipeline | 1.0000 | 1.0000 | 0.0000 | FPs: none, Misses: none |

*(Note: Baseline handles empty input perfectly by extracting nothing, but fails on aliases and negations.)*

## Folder Structure

```text
D:.
├── demo.ps1
├── requirements.txt
├── api
│   ├── main.py
│   └── __init__.py
├── data
│   ├── jds.json
│   ├── resumes.json
│   └── skills_ontology.csv
├── experiments
│   └── training_log.jsonl
├── reports
│   ├── evaluation_results.json
│   └── sign_off_report.md
├── src
│   ├── baseline.py
│   ├── evaluator.py
│   ├── generate_data.py
│   ├── generate_ontology.py
│   ├── model_trainer.py
│   ├── ontology.py
│   ├── pipeline.py
│   ├── report_generator.py
│   ├── rule_extractor.py
│   ├── __init__.py
│   └── models
│       └── skill_classifier.pkl
└── tests
    ├── test_edge_cases.py
    └── __init__.py
```

## Why this isn't just keyword matching

To prove the parser doesn't just guess keywords, the synthetic data was forward-generated with deliberate hard cases: 1% junk/empty text, alias-only phrasing, negation ("no experience with X"), and substring traps ("R" as an initial or inside "Director"). 

Critically, the trained ML confidence filter (Stage 3b) is not decorative — it actively catches false positives that bypass the rule-based candidate generator (Stage 3a). For example, in document `res_substring_01` ("I was a Director of Engineering at R&D corp..."), the Rules-Only layer (Stage 3a) incorrectly extracts both "Data Engineering" (from "Engineering") and "R" (from "R&D"), netting a poor precision of 0.3333. The trained ML model evaluates these candidates, sees the low fuzzy scores or short token flags, assigns them low confidence (e.g., confidence=0.285 for "R"), and correctly filters them out, returning the precision to 1.0000 for that document. This exact behavior is proven in `tests/test_edge_cases.py::TestSubstringFalsePositiveTrap::test_substring_false_positive_trap`.

## Setup & Rebuild Instructions

**1. Create venv and install dependencies**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the pipeline in order**
*(Order matters because the ML model requires the generated data and ontology, and the evaluation report requires the trained model artifact).*
```bash
python src/generate_ontology.py
python src/generate_data.py
python -m src.model_trainer
python -m src.report_generator
python -m pytest tests/test_edge_cases.py -v
```

**3. Start the API**
```bash
uvicorn api.main:app --port 8000
```

## Demo Walkthrough

*(Execute these commands with the API running locally on port 8000. This takes <2 minutes).*

**1. Parse a fresh pasted resume live**
```bash
curl -X POST http://localhost:8000/parse -H "Content-Type: application/json" -d "{\"text\": \"Really love working with React.js and Spring Boot. I have no experience with GraphQL unfortunately.\"}"
```

**2. Explainability Walkthrough (Hits + Misses on a specific document)**
```bash
curl -X GET http://localhost:8000/parse/eval/res_alias_01
```

**3. Edge Case: Negation handling**
```bash
curl -X POST http://localhost:8000/parse -H "Content-Type: application/json" -d "{\"text\": \"I am not familiar with Azure, but I have used AWS before.\"}"
```

**4. Metrics Report (Baseline vs Rules-only vs Full Pipeline)**
```bash
curl -X GET http://localhost:8000/parse/report
```

## Pitfalls Checklist

- [x] **No black-box extractions**: Every extraction includes an `explanation` field detailing match type, location, and the driving features for the model's confidence. (See `src/pipeline.py`).
- [x] **Numbers not vibes**: Precision/recall/FPR computed across the entire synthetic population, not a toy subset. Metrics tracked by `src/evaluator.py`.
- [x] **"Basically fine isn't good enough" (Negation/Substring handling proven)**: Negated skills and substring traps explicitly fail if unhandled; they are tested in `tests/test_edge_cases.py::TestNegationHandling` and `TestSubstringFalsePositiveTrap`.
- [x] **No single blended metric**: Hard cases are broken out in the report separately so failures in negation aren't hidden by high baseline recall on standard text.
- [x] **ML filter proven to fire**: Stage 3a and Stage 3b are distinct. The report explicitly prints precision drops if the Stage 3b model is removed.

## Self-Check Answers

- **Can you show "Parsing v0" working live on arbitrary text, not just the pre-built sample set?** Yes, the `POST /parse` endpoint natively accepts arbitrary text payloads and extracts skills dynamically.
- **What happens on empty/malformed input?** It explicitly returns a safe `no_skills_found` status with an empty skill list (zero confidence guesses). Proven by `tests/test_edge_cases.py::test_empty_malformed_input`.
- **How do we know the skills-ontology dependency is actually being used, not just loaded and ignored?** The `test_alias_only_resume` test specifically runs on a document containing zero canonical names and only aliases/abbreviations. It achieves >50% recall, proving the alias mapping from the ontology is active.
- **Where does Parsing v0 still get it wrong?** Recall drops heavily on alias-only documents (0.7143) because the ML model filters out some valid short aliases (like "R") out of caution. Fuzzy matching can also struggle with novel abbreviations not mapped in the ontology.

*(Note: The e-sign and tamper-evidence self-check questions from the original study guide do not apply to this AI/ML parsing deliverable. E-sign functionality is owned by the Platform/Backend Engineering team. That mismatch is documented here rather than silently ignored.)*

## Hand-off

The final output is a structured profile/job JSON schema, ready to be consumed directly by the next team. The schema includes `doc_id`, `status`, `skills` (list of dictionaries containing `canonical_name`, `confidence`, `match_type`, `matched_text`, `offset_start/end`, and `explanation`), and a `misses` array for evaluation contexts.

**Guardrail Suggestion:** The downstream team should configure an automated alert if the parser's aggregate confidence drops below `0.70` on average for >100 resumes, or if the periodic manual review of live production parses shows `Precision` dropping below `0.90`, triggering a model retraining or ontology update.

## Next Steps

- Expand the hand-labeled ground-truth set using a diverse set of real-world resumes to combat synthetic data bias in the model.
- Replace the synthetic `skills_ontology.csv` with the final, production-ready PlaceMux ontology once finalized by the product team.
- Re-tune the confidence threshold (currently 0.5); if false negatives (missed skills) are heavily penalized in production job matching, we may need to lower the threshold to bias toward recall.
