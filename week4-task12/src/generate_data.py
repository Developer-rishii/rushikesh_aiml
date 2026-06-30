import json
import random
import os

def generate_data():
    os.makedirs('data', exist_ok=True)
    
    # We will generate a mix of normal, and edge-case documents
    resumes = []
    jds = []
    
    # 1. Edge Case: Junk / Irrelevant text
    resumes.append({
        "doc_id": "res_junk_01",
        "text": "I am a very hard worker. I love taking my dog for walks and cooking pasta. My favorite color is blue and I enjoy reading fiction novels on the weekends.",
        "ground_truth": []
    })
    
    # 2. Edge Case: Alias-only resume
    resumes.append({
        "doc_id": "res_alias_01",
        "text": "Senior Engineer with 5 years in ML and DL. Proficient in tf and k8s. I also use r-script and js for quick prototypes. I love working in a dev-ops culture.",
        "ground_truth": ["Machine Learning", "Deep Learning", "TensorFlow", "Kubernetes", "R", "JavaScript", "DevOps"]
    })
    
    # 3. Edge Case: Negation case
    resumes.append({
        "doc_id": "res_negation_01",
        "text": "Experienced Python developer. I have no experience with Docker or Kubernetes. I am not familiar with AWS, but I am learning.",
        "ground_truth": ["Python"]
    })
    
    # 4. Edge Case: Empty / malformed
    jds.append({
        "doc_id": "jd_empty_01",
        "text": "",
        "ground_truth": []
    })
    
    # 5. Edge Case: Substring / false positive trap ("R" the language vs letter R)
    resumes.append({
        "doc_id": "res_substring_01",
        "text": "I was a Director of Engineering at R&D corp. Currently working on a project with Python. I am not a fan of the letter R, but I love programming.",
        "ground_truth": ["Python"]
    })

    # Normal generation templates
    skills_pool = [
        ("Python", "Python"), ("Java", "Java"), ("JavaScript", "JS"), ("C++", "C++"),
        ("Go", "Golang"), ("React", "React"), ("Angular", "Angular"),
        ("Docker", "Docker"), ("Kubernetes", "k8s"), ("AWS", "AWS"),
        ("Machine Learning", "Machine Learning"), ("Data Science", "Data Science"),
        ("Communication", "Communication"), ("Leadership", "Leadership"),
        ("SQL", "SQL"), ("PostgreSQL", "Postgres"), ("MongoDB", "MongoDB")
    ]
    
    templates = [
        "I am a software engineer skilled in {}. I also use {}.",
        "My tech stack includes {}, {}, and {}. I am looking for a new role.",
        "Experienced in {}, with a strong background in {}. Familiar with {}.",
        "Worked extensively with {} and {}. Great {} skills.",
        "Proficient in {}. Used {} in my last project.",
        "{}, {}, {} are my main strengths.",
        "Senior developer: {} | {} | {}"
    ]
    
    jd_templates = [
        "Looking for a software engineer with experience in {}. Must know {}.",
        "Requirements: {}, {}, and {}. Nice to have: {}.",
        "We need an expert in {}. Background in {} is a plus.",
        "Join our team! We use {} and {}. You should have good {}."
    ]
    
    # Generate ~200 resumes
    for i in range(1, 196):
        template = random.choice(templates)
        num_skills = template.count("{}")
        chosen = random.sample(skills_pool, num_skills)
        
        # Format text, decide randomly to use canonical or alias
        formatted_skills = []
        gt = []
        for canonical, alias in chosen:
            gt.append(canonical)
            formatted_skills.append(random.choice([canonical, alias]))
            
        text = template.format(*formatted_skills)
        
        resumes.append({
            "doc_id": f"res_{i:03d}",
            "text": text,
            "ground_truth": gt
        })
        
    # Generate ~80 JDs
    for i in range(1, 80):
        template = random.choice(jd_templates)
        num_skills = template.count("{}")
        chosen = random.sample(skills_pool, num_skills)
        
        formatted_skills = []
        gt = []
        for canonical, alias in chosen:
            gt.append(canonical)
            formatted_skills.append(random.choice([canonical, alias]))
            
        text = template.format(*formatted_skills)
        
        jds.append({
            "doc_id": f"jd_{i:03d}",
            "text": text,
            "ground_truth": gt
        })
        
    with open('data/resumes.json', 'w', encoding='utf-8') as f:
        json.dump(resumes, f, indent=2)
        
    with open('data/jds.json', 'w', encoding='utf-8') as f:
        json.dump(jds, f, indent=2)
        
    print(f"Generated {len(resumes)} resumes and {len(jds)} JDs.")

if __name__ == "__main__":
    generate_data()
