# Parsing → Skills Ontology: Demo Script

This script walks through the live verification requirements for the Skills Ontology Mapper.

## Prerequisites

1. Open a terminal and navigate to the task directory:
   ```bash
   cd d:\Placemux-aiml\week4-task14
   ```
2. Start the API server in one terminal tab:
   ```bash
   python -m src.api
   ```
3. Open a second terminal tab to run the evaluation and live requests.

---

## Step 1: Baseline Metrics

**Goal:** Show the baseline accuracy (exact-match only) to prove what we are measuring against.

1. Run the baseline evaluation:
   ```bash
   python -m src.baseline
   ```
2. **Expected Output:**
   - Accuracy: ~37.78%
   - Recall: ~28.21%
   - "Unmapped detection rate: 100%"

*Note: Explain that this is what happens when we blindly match strings without a synonym layer, fuzzy matching, or semantic similarity.*

## Step 2: Mapper Metrics & Segment Breakdown

**Goal:** Show the layered mapper's numbers, the % improvement vs baseline, and the segment breakdown.

1. Run the full evaluation:
   ```bash
   python -m src.evaluate
   ```
2. **Expected Output:**
   - Overall Accuracy: ~96.30%
   - **Improvement:** ~154.9% relative accuracy improvement
   - Segment Breakdown Table: Prove the mapper wins across abbreviations, multi-word terms, and typos, not just exact matches.

## Step 3: One-Example Walkthrough

**Goal:** Run one live raw skill list through `/map-skills`, showing reasons per term.

1. Run the explainability sanity check script:
   ```bash
   python src/explain.py
   ```
2. **Expected Output:**
   A formatted output showing exactly why each term was mapped:
   - "ReactJS" → React (exact synonym match)
   - "Sr. Java Developer" → Java (semantic similarity, confidence 83%)
   - "•" → UNMAPPED (identified as parsing artifact / noise)

## Step 4: Richer Matching Inputs (The Hand-off)

**Goal:** Call `/match-preview` to show raw vs ontology-tagged skills feeding a match live.

1. Curl the match-preview endpoint:
   ```bash
   curl -s http://localhost:8000/match-preview/stu_001 | python -m json.tool
   ```
2. **Expected Output:**
   A JSON response showing the `raw_skills` alongside the deduplicated `ontology_tagged` list, and a `summary` line proving the hand-off (e.g. "11 raw terms -> 10 unique canonical skills... These 10 clean skills feed the matching engine.").

## Step 5: Graceful Error Handling & Edge Cases

**Goal:** Deliberately send a malformed/empty batch to show graceful error handling.

1. Send an empty list to trigger the 400 validation:
   ```bash
   curl -s -X POST http://localhost:8000/map-skills \
        -H "Content-Type: application/json" \
        -d '{"raw_terms": []}' | python -m json.tool
   ```
2. **Expected Output:**
   A clean `400 Bad Request` with a hint (`Ensure 'raw_terms' is a non-empty list of strings.`), not a 500 error.

3. Send an all-noise payload to prove it won't force wrong matches:
   ```bash
   curl -s -X POST http://localhost:8000/map-skills \
        -H "Content-Type: application/json" \
        -d '{"raw_terms": ["skills:", "N/A", "xkq7z", "---"]}' | python -m json.tool
   ```
4. **Expected Output:**
   `total_mapped: 0`, `total_unmapped: 4`, with plain-English reasons for each unmapped term.

## Step 6: Dependency Status & Mitigation Plan

**Goal:** State current dependency status (Parsing v0) and the mitigation plan.

**Verbal Walkthrough:**
> "We depend on the 'Parsing v0' service for upstream raw skills. Since that is currently unstable/late, I built a realistic synthetic dataset (`data/synthetic_raw_terms.json`) that strictly mimics their schema.
> 
> "Our mitigation plan is documented in `generate_synthetic_raw_terms.py`: If Parsing v0 is late or the schema changes, our loader gracefully logs an error, raises an alert, and continues serving the last-known-good synthetic dataset so downstream matching isn't blocked. We will notify the Parsing v0 team lead within 24 hours rather than failing silently."
