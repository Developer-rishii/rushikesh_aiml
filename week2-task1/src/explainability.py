from typing import Dict, Any, List

def generate_reasons(match_score: int, details: Dict[str, Any]) -> List[str]:
    """
    Translates the raw score details from the matcher into human-readable reasons.
    """
    reasons = []
    
    # 1. Technical Skills
    skills_info = details.get('skills', {})
    for skill, s_details in skills_info.get('breakdown', {}).items():
        diff = s_details['diff']
        if diff > 0:
            reasons.append(f"🟢 {skill} score ({s_details['student_score']}) exceeds requirement ({s_details['required']}) by {diff} points.")
        elif diff == 0:
            reasons.append(f"🟢 {skill} score exactly meets the requirement of {s_details['required']}.")
        else:
            reasons.append(f"🔴 {skill} score ({s_details['student_score']}) falls short of requirement ({s_details['required']}) by {abs(diff)} points.")
            
    # 2. CGPA
    cgpa_info = details.get('cgpa', {})
    cgpa_diff = cgpa_info.get('diff', 0)
    req_cgpa = cgpa_info.get('required', 0)
    student_cgpa = cgpa_info.get('student_cgpa', 0)
    
    if cgpa_diff >= 0:
        reasons.append(f"🟢 CGPA ({student_cgpa}) satisfies the minimum requirement of {req_cgpa}.")
    else:
        reasons.append(f"🔴 CGPA ({student_cgpa}) is below the minimum requirement of {req_cgpa}.")
        
    # 3. Experience
    exp_info = details.get('experience', {})
    req_exp = exp_info.get('required', 0)
    student_units = exp_info.get('student_units', 0)
    internships = exp_info.get('internships', 0)
    projects = exp_info.get('projects', 0)
    
    if student_units >= req_exp:
        if req_exp > 0:
            reasons.append(f"🟢 Experience requirement met with {internships} internship(s) and {projects} project(s).")
    else:
        reasons.append(f"🔴 Lacks required experience. Has {internships} internship(s) and {projects} project(s) vs required {req_exp} units.")
        
    # 4. Soft Skills
    soft_info = details.get('soft_skills', {})
    comm = soft_info.get('comm', 0)
    if comm >= 80:
        reasons.append(f"🟢 Strong communication skills ({comm}).")
    elif comm < 70:
        reasons.append(f"🟡 Communication skills ({comm}) could be improved.")

    return reasons
