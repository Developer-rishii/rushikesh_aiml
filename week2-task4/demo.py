import time
from src.explainability.explainer import ExplainerEngine

def run_demo():
    print("--- PlaceMux Explainability Module Demo ---")
    print("Simulating end-to-end flow...\n")
    
    time.sleep(1)
    
    student = {
        "student_id": 101,
        "skills": "Python,SQL,Machine Learning",
        "experience_years": 1,
        "verified_score": 85
    }
    
    job = {
        "job_id": 501,
        "required_skills": "Python,SQL,Communication",
        "minimum_score": 70
    }
    
    print("1. Student -> Apply")
    print(f"Student {student['student_id']} applied with skills: {student['skills']}, Verified Score: {student['verified_score']}, Experience: {student['experience_years']} years.\n")
    time.sleep(1)
    
    print("2. Job -> Available")
    print(f"Job {job['job_id']} requires skills: {job['required_skills']}, Minimum Score: {job['minimum_score']}.\n")
    time.sleep(1)
    
    print("3. Ranking Module -> Generates Match Score")
    # Simulate a ranking score
    ranking_score = 87
    print(f"Match Score from Ranking Module: {ranking_score}%\n")
    time.sleep(1)
    
    print("4. Explainability Module -> Generates Explanation")
    explainer = ExplainerEngine()
    explanation = explainer.generate_explanation(student, job, match_score=ranking_score)
    print("Explanation JSON Payload Generated.\n")
    time.sleep(1)
    
    print("5. Company Dashboard -> Displays")
    print("-" * 40)
    print(f"Match Score: {explanation['match_score']}%")
    print("\nMatched Skills:")
    for skill in explanation["matched_skills"]:
        print(f"[+] {skill}")
        
    print("\nMissing Skills:")
    for skill in explanation["missing_skills"]:
        print(f"[-] {skill}")
        
    print(f"\nReason:\n{explanation['reason']}")
    print("-" * 40)
    print("\nDemo Completed successfully.")

if __name__ == "__main__":
    run_demo()
