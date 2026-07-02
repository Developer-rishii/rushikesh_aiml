"""
Baseline skill mapper — exact string match only.
Lowercases + strips whitespace, then matches against canonical display_name.
No synonym table, no fuzzy matching.

This establishes the floor that the real mapper must beat.
"""

import json
import os


class BaselineMapper:
    """Maps raw skill terms to canonical skills using exact display_name match only."""

    def __init__(self, ontology_path: str = None):
        if ontology_path is None:
            ontology_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "ontology.json"
            )
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        # Build lookup: lowercased display_name -> skill record
        self.lookup: dict[str, dict] = {}
        for skill in self.ontology:
            key = skill["display_name"].strip().lower()
            self.lookup[key] = skill

    def map_term(self, raw_term: str) -> dict:
        """
        Map a single raw term.
        Returns: {raw, canonical_id, display_name, confidence, method}
        """
        if not raw_term or not isinstance(raw_term, str):
            return {
                "raw": raw_term,
                "canonical_id": "unmapped",
                "display_name": None,
                "confidence": 0.0,
                "method": "invalid_input",
            }

        normalized = raw_term.strip().lower()
        if normalized in self.lookup:
            skill = self.lookup[normalized]
            return {
                "raw": raw_term,
                "canonical_id": skill["canonical_id"],
                "display_name": skill["display_name"],
                "confidence": 1.0,
                "method": "exact_display_name",
            }

        return {
            "raw": raw_term,
            "canonical_id": "unmapped",
            "display_name": None,
            "confidence": 0.0,
            "method": "no_match",
        }

    def map_batch(self, raw_terms: list[str]) -> list[dict]:
        """Map a batch of raw terms."""
        return [self.map_term(t) for t in raw_terms]


def evaluate_baseline(test_set_path: str = None, ontology_path: str = None):
    """Run baseline on test set and print metrics."""
    if test_set_path is None:
        test_set_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test_set.json"
        )

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    mapper = BaselineMapper(ontology_path)

    correct = 0
    total = 0
    false_positives = 0  # mapped to wrong skill
    true_unmapped = 0    # correctly identified as unmapped
    false_unmapped = 0   # should have mapped but didn't

    for record in test_data:
        raw = record["raw_term"]
        expected = record["expected_canonical_id"]
        result = mapper.map_term(raw)

        total += 1
        predicted = result["canonical_id"]

        if predicted == expected:
            correct += 1
            if predicted == "unmapped":
                true_unmapped += 1
        elif predicted != "unmapped" and expected != "unmapped":
            false_positives += 1  # mapped, but to the wrong skill
        elif predicted != "unmapped" and expected == "unmapped":
            false_positives += 1  # mapped something that should be unmapped
        elif predicted == "unmapped" and expected != "unmapped":
            false_unmapped += 1   # missed a real skill

    # Metrics
    precision = correct / total if total > 0 else 0
    mapped_count = sum(1 for r in test_data if mapper.map_term(r["raw_term"])["canonical_id"] != "unmapped")
    mappable_count = sum(1 for r in test_data if r["expected_canonical_id"] != "unmapped")
    recall = (correct - true_unmapped) / mappable_count if mappable_count > 0 else 0
    fp_rate = false_positives / total if total > 0 else 0
    unmapped_total = sum(1 for r in test_data if r["expected_canonical_id"] == "unmapped")
    unmapped_rate = true_unmapped / unmapped_total if unmapped_total > 0 else 0

    print("=" * 60)
    print("BASELINE EVALUATION (exact display_name match only)")
    print("=" * 60)
    print(f"Total test records:       {total}")
    print(f"Correct predictions:      {correct}")
    print(f"Overall accuracy:         {precision:.2%}")
    print(f"Recall (mappable only):   {recall:.2%}")
    print(f"False positive rate:      {fp_rate:.2%}")
    print(f"Unmapped detection rate:  {unmapped_rate:.2%}")
    print(f"  (correctly ID'd {true_unmapped}/{unmapped_total} unmappable terms)")
    print("=" * 60)

    return {
        "total": total,
        "correct": correct,
        "accuracy": precision,
        "recall": recall,
        "false_positive_rate": fp_rate,
        "unmapped_detection_rate": unmapped_rate,
        "false_positives": false_positives,
        "false_unmapped": false_unmapped,
    }


if __name__ == "__main__":
    evaluate_baseline()
