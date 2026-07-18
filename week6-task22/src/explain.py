"""
src/explain.py

For every match/score, produce a plain-English reason (Study Guide
Section 4: "A black box that says 'trust me' is a red flag").

Approach: combine the model's global feature_importances_ with the
*local* feature values for this specific pair, so the explanation is
grounded in this student/this job, not a generic statement.
"""
import json


def explain_match(student_id, job_id, student_skills: dict, job_skills: dict,
                   feature_row: dict, score: float, decision: int, model) -> str:
    overlap = sorted(set(student_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(student_skills))
    top_skill = max(job_skills, key=job_skills.get) if job_skills else None

    importances = model.feature_importance()
    top_driver = max(importances, key=importances.get)

    lines = []
    lines.append(f"Match score for {student_id} -> {job_id}: {score:.2f} "
                  f"(threshold {model.threshold:.2f}) -> "
                  f"{'RECOMMENDED' if decision else 'NOT RECOMMENDED'}.")

    if overlap:
        lines.append(f"Shared skills ({len(overlap)}/{len(job_skills)} of the JD): {', '.join(overlap)}.")
    else:
        lines.append("No skill overlap with this JD at all.")

    if top_skill:
        if top_skill in student_skills:
            lines.append(f"Has the JD's top-priority skill '{top_skill}' "
                          f"(verified score {student_skills[top_skill]:.2f}).")
        else:
            lines.append(f"Missing the JD's top-priority skill '{top_skill}' -- "
                          f"this alone drags the score down significantly.")

    if feature_row.get("years_gap", 0) < 0:
        lines.append(f"Has {abs(feature_row['years_gap'])} fewer year(s) of experience than the JD asks for.")
    elif feature_row.get("years_gap", 0) > 0:
        lines.append(f"Exceeds the JD's experience requirement by {feature_row['years_gap']} year(s).")

    if missing:
        shown = ", ".join(missing[:4]) + ("..." if len(missing) > 4 else "")
        lines.append(f"Gaps to close: {shown}.")

    lines.append(f"Biggest driver of this model's decisions in general: '{top_driver}' "
                 f"(importance {importances[top_driver]:.2f}).")

    return " ".join(lines)
