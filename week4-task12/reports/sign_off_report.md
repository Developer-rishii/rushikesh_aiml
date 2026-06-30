# Parsing v0 — Sign-Off Report

*Generated: 2026-06-30T15:34:42.629417*

## 1. What "Good" Looks Like

Parsing v0 takes unstructured resume or job description text and extracts **structured skills**
normalized against a canonical skills ontology. Success means:
- End-to-end demoable on arbitrary text (not just pre-built samples)
- Real precision/recall/FPR numbers computed on held-out data
- A trained ML model that measurably improves over raw rule-based extraction
- Honest handling of edge cases (negation, junk text, substring traps, aliases)

## 2. Upstream Dependency: Skills Ontology

- **File**: `data/skills_ontology.csv`
- **Size**: 66 canonical skills
- **Categories**: {"language": 14, "concept": 14, "framework": 11, "tool": 8, "database": 6, "library": 5, "soft-skill": 5, "platform": 3}
- **Structure**: `canonical_name`, `category`, `aliases` (pipe-separated synonyms/abbreviations)
- **Validation test**: `tests/test_edge_cases.py::test_ontology_validation_malformed` — confirms the pipeline fails loudly if the ontology is malformed (missing columns, duplicates, empty aliases)

## 3. Baseline Numbers (Raw Keyword Match)

| Segment | Precision | Recall | FPR | TP | FP | FN |
|---------|-----------|--------|-----|----|----|-----|
| Resume | 0.9917 | 0.8782 | 0.0083 | 476 | 4 | 66 |
| Jd | 1.0000 | 0.8894 | 0.0000 | 193 | 0 | 24 |
| Overall | 0.9941 | 0.8814 | 0.0059 | 669 | 4 | 90 |

## 4. Trained ML Model

- **Model type**: RandomForestClassifier
- **Parameters**: {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced"}
- **Training timestamp**: 2026-06-30T15:27:46.742783
- **Data splits**: Train=515, Val=131, Test=281
- **Saved artifact**: `src/models/skill_classifier.pkl`
- **Train accuracy**: 0.9961
- **Val accuracy**: 1.0000, Precision: 1.0000, Recall: 1.0000
- **Test accuracy**: 0.9893, Precision: 0.9868, Recall: 1.0000
- **Feature importances**: {"match_exact": 0.2421, "match_alias": 0.0245, "match_fuzzy": 0.3781, "fuzzy_score": 0.2662, "is_negated": 0.0, "is_short_token": 0.0071, "token_length": 0.082}

**Labeling logic**: A candidate from Stage 3a is labeled true-positive (1) if its `canonical_name` appears in the document's hand-labeled ground-truth skill list; otherwise it is labeled false-positive (0). Training labels are derived from forward-generated ground truth (the skills embedded when generating synthetic data), NOT from whatever the parser happens to find.

## 5. Parsing v0 (Full Pipeline) vs Baseline vs Rules-Only

### Overall Comparison

| Method | Precision | Recall | FPR | TP | FP | FN |
|--------|-----------|--------|-----|----|----|-----|
| Baseline | 0.9941 | 0.8814 | 0.0059 | 669 | 4 | 90 |
| Rules-Only (3a) | 0.8203 | 0.9987 | 0.1797 | 758 | 166 | 1 |
| Full Pipeline (3a+3b) | 0.9987 | 0.9974 | 0.0013 | 757 | 1 | 2 |

### By Segment: Resumes

| Method | Precision | Recall | FPR |
|--------|-----------|--------|-----|
| Baseline | 0.9917 | 0.8782 | 0.0083 |
| Rules-Only | 0.8111 | 0.9982 | 0.1889 |
| Full Pipeline | 0.9982 | 0.9963 | 0.0018 |

### By Segment: JDs

| Method | Precision | Recall | FPR |
|--------|-----------|--------|-----|
| Baseline | 1.0000 | 0.8894 | 0.0000 |
| Rules-Only | 0.8444 | 1.0000 | 0.1556 |
| Full Pipeline | 1.0000 | 1.0000 | 0.0000 |

### Hard-Case Breakdown

**Junk/irrelevant text (no skills)**

| Method | Precision | Recall | FPR | Details |
|--------|-----------|--------|-----|---------|
| Baseline | 0.0000 | 0.0000 | 0.0000 | FPs: none, Misses: none |
| Rules-Only | 0.0000 | 0.0000 | 1.0000 | FPs: ['Leadership'], Misses: none |
| Full Pipeline | 0.0000 | 0.0000 | 0.0000 | FPs: none, Misses: none |

**Alias-only resume (no canonical names)**

| Method | Precision | Recall | FPR | Details |
|--------|-----------|--------|-----|---------|
| Baseline | 1.0000 | 0.1429 | 0.0000 | FPs: none, Misses: ['TensorFlow', 'JavaScript', 'Machine Learning', 'Deep Learning', 'Kubernetes', 'DevOps'] |
| Rules-Only | 0.8571 | 0.8571 | 0.1429 | FPs: ['Terraform'], Misses: ['TensorFlow'] |
| Full Pipeline | 0.8333 | 0.7143 | 0.1667 | FPs: ['Terraform'], Misses: ['TensorFlow', 'R'] |

**Negation case ("no experience with X")**

| Method | Precision | Recall | FPR | Details |
|--------|-----------|--------|-----|---------|
| Baseline | 0.2500 | 1.0000 | 0.7500 | FPs: ['Docker', 'Kubernetes', 'AWS'], Misses: none |
| Rules-Only | 0.5000 | 1.0000 | 0.5000 | FPs: ['Leadership'], Misses: none |
| Full Pipeline | 1.0000 | 1.0000 | 0.0000 | FPs: none, Misses: none |

**Empty/malformed input**

| Method | Precision | Recall | FPR | Details |
|--------|-----------|--------|-----|---------|
| Baseline | 0.0000 | 0.0000 | 0.0000 | FPs: none, Misses: none |
| Rules-Only | 0.0000 | 0.0000 | 0.0000 | FPs: none, Misses: none |
| Full Pipeline | 0.0000 | 0.0000 | 0.0000 | FPs: none, Misses: none |

**Substring trap (e.g. "R" in "Director")**

| Method | Precision | Recall | FPR | Details |
|--------|-----------|--------|-----|---------|
| Baseline | 0.5000 | 1.0000 | 0.5000 | FPs: ['R'], Misses: none |
| Rules-Only | 0.3333 | 1.0000 | 0.6667 | FPs: ['Data Engineering', 'R'], Misses: none |
| Full Pipeline | 1.0000 | 1.0000 | 0.0000 | FPs: none, Misses: none |

### Verdict

The full pipeline (precision=0.9987) **improves precision** over the unfiltered rule-based extractor (precision=0.8203).
However, recall drops from 0.9987 to 0.9974 due to the model filtering some true positives.

**Known limitations of v0:**
- Fuzzy matching may miss skills with unusual abbreviations not in the ontology
- Negation detection uses a fixed set of cue phrases; sarcasm or complex negation structures may not be caught
- Short/ambiguous tokens (e.g. 'R', 'Go', 'C') require careful boundary detection and may still have edge cases
- The model is trained on synthetic data; real-world resumes may have different distributions

## 6. Worked Examples

### Example: Clean Hit
**Document**: `res_001`

**Text**: "Worked extensively with SQL and Java. Great Machine Learning skills."

**Ground truth**: ['SQL', 'Java', 'Machine Learning']

**Extracted skills**:

- Extracted 'SQL': matched canonical name 'SQL' at offset 24; model confidence 0.99, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'Machine Learning': matched canonical name 'Machine Learning' at offset 44; model confidence 1.00, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'Java': matched canonical name 'Java' at offset 32; model confidence 0.99, driven mainly by match_type=not fuzzy and fuzzy_score=1.00

### Example: With Miss
**Document**: `res_alias_01`

**Text**: "Senior Engineer with 5 years in ML and DL. Proficient in tf and k8s. I also use r-script and js for quick prototypes. I love working in a dev-ops culture."

**Ground truth**: ['Machine Learning', 'Deep Learning', 'TensorFlow', 'Kubernetes', 'R', 'JavaScript', 'DevOps']

**Extracted skills**:

- Extracted 'JavaScript': matched alias 'js' at offset 93; model confidence 0.70, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'Terraform': matched alias 'tf' at offset 57; model confidence 0.70, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'Kubernetes': matched alias 'k8s' at offset 64; model confidence 1.00, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'Machine Learning': matched alias 'ML' at offset 32; model confidence 0.70, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'Deep Learning': matched alias 'DL' at offset 39; model confidence 0.70, driven mainly by match_type=not fuzzy and fuzzy_score=1.00
- Extracted 'DevOps': matched alias 'dev-ops' at offset 138; model confidence 1.00, driven mainly by match_type=not fuzzy and fuzzy_score=1.00

**Missed skills**:

- `TensorFlow`: not_found_in_text
- `R`: filtered_by_model_low_confidence (confidence=0.285)

## 7. Edge Cases Tested (Section 5)

| Edge Case | Test Name | What It Proves |
|-----------|-----------|----------------|
| Ontology validation | `test_ontology_validation_malformed` | Pipeline fails loudly with clear error on malformed ontology |
| Empty/malformed input | `test_empty_malformed_input` | Returns explicit `no_skills_found` with zero confidence, not an error |
| Negation handling | `test_negation_handling` | "no experience with Docker" does NOT extract Docker |
| Substring trap | `test_substring_false_positive_trap` | 'R' inside 'Director' or 'R&D' is not falsely extracted |
| Alias-only resume | `test_alias_only_resume` | Resume using only abbreviations still achieves reasonable recall |

## 8. E-Sign / Tamper-Evidence Scope Statement

E-sign integration and tamper-evidence verification are **out of scope** for this AI/ML parsing slice. This week's team-wide theme includes e-signing, but the deliverable for the AI/ML engineer role is "Parsing v0" — structured skill extraction from unstructured text. E-sign functionality (offer letter signing, cryptographic tamper-evidence on signed documents) is owned by the **Platform/Backend Engineering team** and is their separate deliverable this sprint. The study guide's self-check questions about e-signing pertain to that team's work, not this slice.

## 9. Hand-Off Schema

The output of Parsing v0 is a **structured profile/job** JSON object. The next team can consume it directly.

```json
{
  "doc_id": "string \u2014 unique document identifier",
  "doc_type": "string \u2014 'resume' or 'jd'",
  "status": "string \u2014 'ok' or 'no_skills_found'",
  "skills": [
    {
      "canonical_name": "string \u2014 normalized skill name from ontology",
      "confidence": "float \u2014 model confidence (0.0\u20131.0)",
      "match_type": "string \u2014 'exact', 'alias', or 'fuzzy'",
      "matched_text": "string \u2014 the exact text span that triggered the match",
      "offset_start": "int \u2014 character offset of match start",
      "offset_end": "int \u2014 character offset of match end",
      "explanation": "string \u2014 plain-English reason for extraction"
    }
  ],
  "misses": [
    {
      "canonical_name": "string \u2014 ground-truth skill not extracted",
      "reason": "string \u2014 'not_found_in_text', 'filtered_by_negation', or 'filtered_by_model_low_confidence'"
    }
  ]
}
```

## 10. Self-Check

1. **Can Parsing v0 be shown working live on arbitrary text?** Yes — the `POST /parse` endpoint accepts any pasted resume/JD text and returns structured skills with explanations. It does not require the input to be from the pre-built sample set.
2. **What happens on empty/malformed input?** An explicit `no_skills_found` result with zero extracted skills is returned — not an error, not a guess. Proven by `test_empty_malformed_input`.
3. **How do we know the ontology dependency is actually being used?** The `test_alias_only_resume` test uses a resume containing ONLY abbreviations/aliases (no canonical names). If the ontology's alias mapping wasn't being used, recall would be 0%. The test asserts reasonable recall, directly proving the ontology is active.
4. **Where does Parsing v0 still get it wrong?** See the hard-case breakdown above. Known limitations include edge cases in fuzzy matching, complex negation, and short-token disambiguation. These are stated openly, not hidden.
