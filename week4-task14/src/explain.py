"""
Explainability module — generates plain-English reasons for every mapping.

Each mapping result from the mapper includes a 'method' and 'match_detail' field.
This module converts those into human-readable explanations.
"""

import json
import os


def generate_reason(mapping_result: dict) -> str:
    """
    Generate a plain-English reason string for a mapping result.

    Args:
        mapping_result: dict with keys: raw, canonical_id, display_name,
                        confidence, method, match_detail

    Returns:
        A human-readable reason string.
    """
    raw = mapping_result.get("raw", "???")
    canonical_id = mapping_result.get("canonical_id", "unmapped")
    display_name = mapping_result.get("display_name")
    confidence = mapping_result.get("confidence", 0.0)
    method = mapping_result.get("method", "unknown")
    detail = mapping_result.get("match_detail", "")

    if canonical_id == "unmapped":
        # Unmapped — explain why
        if method == "invalid_input":
            return f"Not mapped: '{raw}' — empty or invalid input."
        elif method == "noise_filter":
            return f"Not mapped: '{raw}' — {detail}."
        elif method == "empty_after_preprocess":
            return f"Not mapped: '{raw}' — {detail}."
        elif method == "no_confident_match":
            return f"Not mapped: '{raw}' — {detail}."
        else:
            return f"Not mapped: '{raw}' — no confident match found."

    # Mapped — explain how
    if method == "exact_synonym_lookup":
        return (
            f"Mapped '{raw}' → {display_name}: "
            f"exact synonym match ({detail}), confidence {confidence:.0%}."
        )
    elif method == "exact_display_name":
        return (
            f"Mapped '{raw}' → {display_name}: "
            f"exact display name match, confidence {confidence:.0%}."
        )
    elif method == "fuzzy_match":
        return (
            f"Mapped '{raw}' → {display_name}: "
            f"{detail}, confidence {confidence:.0%}."
        )
    elif method == "tfidf_cosine":
        return (
            f"Mapped '{raw}' → {display_name}: "
            f"semantic similarity ({detail}), confidence {confidence:.0%}."
        )
    else:
        return (
            f"Mapped '{raw}' → {display_name}: "
            f"method={method}, confidence {confidence:.0%}."
        )


def one_example_walkthrough(mapper, raw_skills: list[str] = None) -> str:
    """
    Run a single real-shaped raw resume skill list through the mapper and
    produce a formatted walkthrough showing each term → mapping → reason.

    Args:
        mapper: SkillMapper instance
        raw_skills: Optional list of raw skill strings.
                    If None, uses a realistic built-in example.

    Returns:
        Formatted string showing the full walkthrough.
    """
    if raw_skills is None:
        raw_skills = [
            "Python (3 yrs)",
            "ReactJS",
            "Sr. Java Developer",
            "machine learning",
            "K8s",
            "postgress",
            "agile",
            "•",
            "skills:",
            "Team leadership",
            "Docker Compose",
            "TensorFlow",
            "pytohn",
            "xkq7z",
            "flutter",
        ]

    lines = []
    lines.append("=" * 75)
    lines.append("ONE-EXAMPLE WALKTHROUGH: Raw Resume → Ontology Mapping")
    lines.append("=" * 75)
    lines.append(f"\nInput: {len(raw_skills)} raw parsed skill terms from a resume\n")

    mapped_results = []
    unmapped_results = []

    for raw in raw_skills:
        result = mapper.map_term(raw)
        reason = generate_reason(result)

        if result["canonical_id"] == "unmapped":
            unmapped_results.append(result)
            status = "❌ UNMAPPED"
        else:
            mapped_results.append(result)
            status = f"✅ → {result['display_name']}"

        lines.append(f"  Raw: \"{raw}\"")
        lines.append(f"    {status}")
        lines.append(f"    Reason: {reason}")
        lines.append("")

    lines.append("-" * 75)
    lines.append(f"Summary: {len(mapped_results)} mapped, {len(unmapped_results)} unmapped")
    lines.append(f"Mapped skills: {[r['display_name'] for r in mapped_results]}")
    lines.append(f"Unmapped terms: {[r['raw'] for r in unmapped_results]}")
    lines.append("=" * 75)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from mapper import SkillMapper

    mapper = SkillMapper()
    print(one_example_walkthrough(mapper))
