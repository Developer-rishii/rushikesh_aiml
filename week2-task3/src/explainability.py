import pandas as pd

def generate_explanation(student_row, job_row, score):
    """
    Generates human-readable explainability output for a student-job match.
    """
    s_skills = set(student_row['skills'].split(", ")) if pd.notna(student_row['skills']) else set()
    j_skills = set(job_row['required_skills'].split(", ")) if pd.notna(job_row['required_skills']) else set()
    
    reasons = []
    
    # Skills match
    for skill in j_skills:
        if skill in s_skills:
            reasons.append(f"✓ {skill} matched")
        else:
            reasons.append(f"✗ {skill} skill missing")
            
    # Experience match
    if student_row['experience_years'] >= job_row['min_experience']:
        reasons.append("✓ Experience requirement satisfied")
    else:
        reasons.append(f"✗ Experience requirement not met (needs {job_row['min_experience']}, has {student_row['experience_years']})")
        
    # Education match
    edu_score = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
    s_edu_val = edu_score.get(student_row['education'], 0)
    j_edu_val = edu_score.get(job_row['education_required'], 0)
    
    if s_edu_val >= j_edu_val:
        reasons.append("✓ Education requirement satisfied")
    else:
        reasons.append(f"✗ Education requirement not met (needs {job_row['education_required']})")
        
    # Location match
    if student_row['location'] == job_row['location'] or job_row['location'] == 'Remote':
        reasons.append("✓ Location match satisfied")
    else:
        reasons.append("✗ Location mismatch")
        
    return {
        "match_score": round(score * 100),
        "reasons": reasons
    }

if __name__ == "__main__":
    # Test it with a random pair
    students = pd.read_csv("data/students.csv")
    jobs = pd.read_csv("data/jobs.csv")
    
    explanation = generate_explanation(students.iloc[0], jobs.iloc[0], 0.89)
    print(f"Match Score: {explanation['match_score']}%")
    print("Reasons:")
    for r in explanation['reasons']:
        print(r)
