import csv
import os

def create_ontology():
    os.makedirs('data', exist_ok=True)
    ontology_path = 'data/skills_ontology.csv'
    
    # Structure: canonical_name, category, aliases (pipe-separated)
    skills = [
        # Languages
        ("Python", "language", "python|python3|py"),
        ("Java", "language", "java|core java|j2ee"),
        ("JavaScript", "language", "javascript|js|ecmascript"),
        ("TypeScript", "language", "typescript|ts"),
        ("C++", "language", "c++|cpp|c plus plus"),
        ("C#", "language", "c#|csharp|.net c#"),
        ("Go", "language", "go|golang"),
        ("Rust", "language", "rust|rustlang"),
        ("Ruby", "language", "ruby|rb"),
        ("PHP", "language", "php"),
        ("Swift", "language", "swift"),
        ("Kotlin", "language", "kotlin|kt"),
        ("R", "language", "r|r-script|r programming"),
        ("SQL", "language", "sql|structured query language"),
        
        # Frameworks & Libraries
        ("React", "framework", "react|reactjs|react.js"),
        ("Angular", "framework", "angular|angularjs|angular.js"),
        ("Vue.js", "framework", "vue|vuejs|vue.js"),
        ("Django", "framework", "django"),
        ("Flask", "framework", "flask"),
        ("FastAPI", "framework", "fastapi|fast-api"),
        ("Spring Boot", "framework", "spring|springboot|spring-boot"),
        ("Express.js", "framework", "express|expressjs|express.js"),
        ("Node.js", "framework", "node|nodejs|node.js"),
        ("Ruby on Rails", "framework", "rails|ror"),
        ("Laravel", "framework", "laravel"),
        ("Pandas", "library", "pandas|pd"),
        ("NumPy", "library", "numpy|np"),
        ("Scikit-Learn", "library", "scikit-learn|sklearn|scikit"),
        ("TensorFlow", "library", "tensorflow|tf"),
        ("PyTorch", "library", "pytorch|torch"),
        
        # Tools & Platforms
        ("Git", "tool", "git|github|gitlab|bitbucket"),
        ("Docker", "tool", "docker|dockerize|containerization"),
        ("Kubernetes", "tool", "kubernetes|k8s"),
        ("AWS", "platform", "aws|amazon web services"),
        ("Azure", "platform", "azure|microsoft azure"),
        ("Google Cloud", "platform", "gcp|google cloud platform"),
        ("Linux", "tool", "linux|ubuntu|debian|centos"),
        ("Jenkins", "tool", "jenkins|ci/cd"),
        ("Terraform", "tool", "terraform|tf"),
        ("Ansible", "tool", "ansible"),
        ("Jira", "tool", "jira|atlassian jira"),
        
        # Domains / Concepts
        ("Machine Learning", "concept", "machine learning|ml|machine-learning"),
        ("Deep Learning", "concept", "deep learning|dl"),
        ("Data Science", "concept", "data science|ds"),
        ("Natural Language Processing", "concept", "natural language processing|nlp"),
        ("Computer Vision", "concept", "computer vision|cv"),
        ("Artificial Intelligence", "concept", "artificial intelligence|ai"),
        ("Data Engineering", "concept", "data engineering"),
        ("DevOps", "concept", "devops|dev-ops"),
        ("Agile", "concept", "agile|scrum|kanban"),
        ("Object-Oriented Programming", "concept", "oop|object oriented programming"),
        ("REST API", "concept", "rest|rest api|restful|restful api"),
        ("GraphQL", "concept", "graphql|graph ql"),
        ("Microservices", "concept", "microservices|micro-services|micro service"),
        ("CI/CD", "concept", "ci/cd|continuous integration|continuous deployment"),
        
        # Databases
        ("MySQL", "database", "mysql|my sql"),
        ("PostgreSQL", "database", "postgresql|postgres|postgre sql"),
        ("MongoDB", "database", "mongodb|mongo|mongo db"),
        ("Redis", "database", "redis"),
        ("Cassandra", "database", "cassandra"),
        ("Elasticsearch", "database", "elasticsearch|elastic search"),
        
        # Soft Skills
        ("Communication", "soft-skill", "communication|verbal communication|written communication"),
        ("Leadership", "soft-skill", "leadership|leading|team lead"),
        ("Problem Solving", "soft-skill", "problem solving|problem-solving"),
        ("Teamwork", "soft-skill", "teamwork|collaboration|team player"),
        ("Time Management", "soft-skill", "time management")
    ]
    
    with open(ontology_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['canonical_name', 'category', 'aliases'])
        for skill in skills:
            writer.writerow(skill)
            
    print(f"Ontology generated with {len(skills)} skills at {ontology_path}")

if __name__ == "__main__":
    create_ontology()
