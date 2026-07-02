import json
import os

def build_ontology():
    ontology = [
        # Languages
        {"canonical_id": "skill_python", "display_name": "Python", "category": "Language", "synonyms": ["python", "python3", "python 3", "py"]},
        {"canonical_id": "skill_javascript", "display_name": "JavaScript", "category": "Language", "synonyms": ["javascript", "js", "ecmascript", "es6"]},
        {"canonical_id": "skill_java", "display_name": "Java", "category": "Language", "synonyms": ["java", "java 8", "java 11", "core java"]},
        {"canonical_id": "skill_csharp", "display_name": "C#", "category": "Language", "synonyms": ["c#", "csharp", "c sharp", ".net c#"]},
        {"canonical_id": "skill_cpp", "display_name": "C++", "category": "Language", "synonyms": ["c++", "cpp", "c plus plus"]},
        {"canonical_id": "skill_typescript", "display_name": "TypeScript", "category": "Language", "synonyms": ["typescript", "ts"]},
        {"canonical_id": "skill_go", "display_name": "Go", "category": "Language", "synonyms": ["go", "golang"]},
        {"canonical_id": "skill_ruby", "display_name": "Ruby", "category": "Language", "synonyms": ["ruby", "ruby mri"]},
        {"canonical_id": "skill_php", "display_name": "PHP", "category": "Language", "synonyms": ["php", "php7", "php8"]},
        {"canonical_id": "skill_swift", "display_name": "Swift", "category": "Language", "synonyms": ["swift", "swift ui", "swiftui"]},
        {"canonical_id": "skill_kotlin", "display_name": "Kotlin", "category": "Language", "synonyms": ["kotlin", "kt"]},
        {"canonical_id": "skill_rust", "display_name": "Rust", "category": "Language", "synonyms": ["rust", "rustlang"]},
        {"canonical_id": "skill_scala", "display_name": "Scala", "category": "Language", "synonyms": ["scala"]},
        {"canonical_id": "skill_sql", "display_name": "SQL", "category": "Language", "synonyms": ["sql", "structured query language"]},
        {"canonical_id": "skill_html", "display_name": "HTML", "category": "Language", "synonyms": ["html", "html5"]},
        {"canonical_id": "skill_css", "display_name": "CSS", "category": "Language", "synonyms": ["css", "css3"]},
        {"canonical_id": "skill_bash", "display_name": "Bash", "category": "Language", "synonyms": ["bash", "shell scripting", "shell", "sh"]},
        {"canonical_id": "skill_r", "display_name": "R", "category": "Language", "synonyms": ["r", "r programming", "rlang"]},
        {"canonical_id": "skill_dart", "display_name": "Dart", "category": "Language", "synonyms": ["dart", "dartlang"]},
        {"canonical_id": "skill_objectivec", "display_name": "Objective-C", "category": "Language", "synonyms": ["objective-c", "obj-c", "objc"]},
        
        # Frameworks & Libraries
        {"canonical_id": "skill_react", "display_name": "React", "category": "Framework", "synonyms": ["react", "react.js", "reactjs", "react js"]},
        {"canonical_id": "skill_angular", "display_name": "Angular", "category": "Framework", "synonyms": ["angular", "angularjs", "angular.js", "angular 2+"]},
        {"canonical_id": "skill_vue", "display_name": "Vue.js", "category": "Framework", "synonyms": ["vue", "vuejs", "vue.js", "vue js"]},
        {"canonical_id": "skill_django", "display_name": "Django", "category": "Framework", "synonyms": ["django", "django rest framework", "drf"]},
        {"canonical_id": "skill_flask", "display_name": "Flask", "category": "Framework", "synonyms": ["flask"]},
        {"canonical_id": "skill_fastapi", "display_name": "FastAPI", "category": "Framework", "synonyms": ["fastapi", "fast api"]},
        {"canonical_id": "skill_spring_boot", "display_name": "Spring Boot", "category": "Framework", "synonyms": ["spring boot", "springboot", "spring"]},
        {"canonical_id": "skill_aspnet", "display_name": "ASP.NET", "category": "Framework", "synonyms": ["asp.net", "aspnet", ".net", "dotnet"]},
        {"canonical_id": "skill_express", "display_name": "Express.js", "category": "Framework", "synonyms": ["express", "expressjs", "express.js"]},
        {"canonical_id": "skill_rubyonrails", "display_name": "Ruby on Rails", "category": "Framework", "synonyms": ["ruby on rails", "rails", "ror"]},
        {"canonical_id": "skill_laravel", "display_name": "Laravel", "category": "Framework", "synonyms": ["laravel"]},
        {"canonical_id": "skill_nextjs", "display_name": "Next.js", "category": "Framework", "synonyms": ["nextjs", "next.js", "next js"]},
        {"canonical_id": "skill_react_native", "display_name": "React Native", "category": "Framework", "synonyms": ["react native", "react-native", "rn"]},
        {"canonical_id": "skill_flutter", "display_name": "Flutter", "category": "Framework", "synonyms": ["flutter"]},
        {"canonical_id": "skill_pandas", "display_name": "Pandas", "category": "Library", "synonyms": ["pandas", "pd"]},
        {"canonical_id": "skill_numpy", "display_name": "NumPy", "category": "Library", "synonyms": ["numpy", "np"]},
        {"canonical_id": "skill_scikit_learn", "display_name": "Scikit-Learn", "category": "Library", "synonyms": ["scikit-learn", "scikit learn", "sklearn"]},
        {"canonical_id": "skill_tensorflow", "display_name": "TensorFlow", "category": "Library", "synonyms": ["tensorflow", "tf"]},
        {"canonical_id": "skill_pytorch", "display_name": "PyTorch", "category": "Library", "synonyms": ["pytorch", "torch"]},
        {"canonical_id": "skill_keras", "display_name": "Keras", "category": "Library", "synonyms": ["keras"]},
        {"canonical_id": "skill_tailwind", "display_name": "Tailwind CSS", "category": "Library", "synonyms": ["tailwind", "tailwindcss", "tailwind css"]},
        {"canonical_id": "skill_bootstrap", "display_name": "Bootstrap", "category": "Library", "synonyms": ["bootstrap"]},

        # Tools & Platforms
        {"canonical_id": "skill_git", "display_name": "Git", "category": "Tool", "synonyms": ["git", "github", "gitlab", "bitbucket"]},
        {"canonical_id": "skill_docker", "display_name": "Docker", "category": "Tool", "synonyms": ["docker", "dockerfile", "docker compose"]},
        {"canonical_id": "skill_kubernetes", "display_name": "Kubernetes", "category": "Tool", "synonyms": ["kubernetes", "k8s"]},
        {"canonical_id": "skill_aws", "display_name": "AWS", "category": "Platform", "synonyms": ["aws", "amazon web services"]},
        {"canonical_id": "skill_gcp", "display_name": "Google Cloud Platform", "category": "Platform", "synonyms": ["gcp", "google cloud"]},
        {"canonical_id": "skill_azure", "display_name": "Microsoft Azure", "category": "Platform", "synonyms": ["azure", "microsoft azure"]},
        {"canonical_id": "skill_terraform", "display_name": "Terraform", "category": "Tool", "synonyms": ["terraform", "tf"]},
        {"canonical_id": "skill_jenkins", "display_name": "Jenkins", "category": "Tool", "synonyms": ["jenkins", "ci/cd"]},
        {"canonical_id": "skill_linux", "display_name": "Linux", "category": "Platform", "synonyms": ["linux", "ubuntu", "centos", "unix"]},
        {"canonical_id": "skill_jira", "display_name": "Jira", "category": "Tool", "synonyms": ["jira", "atlassian jira"]},
        {"canonical_id": "skill_postgresql", "display_name": "PostgreSQL", "category": "Database", "synonyms": ["postgresql", "postgres"]},
        {"canonical_id": "skill_mysql", "display_name": "MySQL", "category": "Database", "synonyms": ["mysql"]},
        {"canonical_id": "skill_mongodb", "display_name": "MongoDB", "category": "Database", "synonyms": ["mongodb", "mongo"]},
        {"canonical_id": "skill_redis", "display_name": "Redis", "category": "Database", "synonyms": ["redis"]},
        {"canonical_id": "skill_elasticsearch", "display_name": "Elasticsearch", "category": "Database", "synonyms": ["elasticsearch", "elastic search", "es"]},
        {"canonical_id": "skill_kafka", "display_name": "Apache Kafka", "category": "Tool", "synonyms": ["kafka", "apache kafka"]},
        {"canonical_id": "skill_rabbitmq", "display_name": "RabbitMQ", "category": "Tool", "synonyms": ["rabbitmq"]},
        {"canonical_id": "skill_graphql", "display_name": "GraphQL", "category": "Tool", "synonyms": ["graphql"]},
        {"canonical_id": "skill_rest_api", "display_name": "REST APIs", "category": "Tool", "synonyms": ["rest", "rest api", "restful api", "restful services"]},

        # Soft Skills & Domains
        {"canonical_id": "skill_machine_learning", "display_name": "Machine Learning", "category": "Domain", "synonyms": ["machine learning", "ml"]},
        {"canonical_id": "skill_deep_learning", "display_name": "Deep Learning", "category": "Domain", "synonyms": ["deep learning", "dl"]},
        {"canonical_id": "skill_nlp", "display_name": "NLP", "category": "Domain", "synonyms": ["nlp", "natural language processing", "text mining"]},
        {"canonical_id": "skill_computer_vision", "display_name": "Computer Vision", "category": "Domain", "synonyms": ["computer vision", "cv", "image processing"]},
        {"canonical_id": "skill_data_engineering", "display_name": "Data Engineering", "category": "Domain", "synonyms": ["data engineering", "etl", "data pipelines"]},
        {"canonical_id": "skill_data_science", "display_name": "Data Science", "category": "Domain", "synonyms": ["data science", "data analytics"]},
        {"canonical_id": "skill_agile", "display_name": "Agile Methodology", "category": "Soft Skill", "synonyms": ["agile", "scrum", "kanban"]},
        {"canonical_id": "skill_leadership", "display_name": "Leadership", "category": "Soft Skill", "synonyms": ["leadership", "team leading", "management"]},
        {"canonical_id": "skill_communication", "display_name": "Communication", "category": "Soft Skill", "synonyms": ["communication", "verbal communication", "written communication"]},
        {"canonical_id": "skill_problem_solving", "display_name": "Problem Solving", "category": "Soft Skill", "synonyms": ["problem solving", "analytical skills"]},
        {"canonical_id": "skill_system_design", "display_name": "System Design", "category": "Domain", "synonyms": ["system design", "software architecture", "architecture"]},
        {"canonical_id": "skill_cloud_computing", "display_name": "Cloud Computing", "category": "Domain", "synonyms": ["cloud computing", "cloud infrastructure"]},
        {"canonical_id": "skill_devops", "display_name": "DevOps", "category": "Domain", "synonyms": ["devops", "site reliability engineering", "sre"]},
        {"canonical_id": "skill_cybersecurity", "display_name": "Cybersecurity", "category": "Domain", "synonyms": ["cybersecurity", "security", "infosec", "information security"]},
        {"canonical_id": "skill_blockchain", "display_name": "Blockchain", "category": "Domain", "synonyms": ["blockchain", "web3", "smart contracts", "crypto"]},
        {"canonical_id": "skill_ui_ux", "display_name": "UI/UX Design", "category": "Domain", "synonyms": ["ui", "ux", "ui/ux", "user interface", "user experience"]},
        {"canonical_id": "skill_project_management", "display_name": "Project Management", "category": "Domain", "synonyms": ["project management", "pm", "pmp"]}
    ]

    more_skills = [
        {"canonical_id": "skill_matlab", "display_name": "MATLAB", "category": "Language", "synonyms": ["matlab"]},
        {"canonical_id": "skill_vba", "display_name": "VBA", "category": "Language", "synonyms": ["vba", "visual basic for applications"]},
        {"canonical_id": "skill_haskell", "display_name": "Haskell", "category": "Language", "synonyms": ["haskell"]},
        {"canonical_id": "skill_lua", "display_name": "Lua", "category": "Language", "synonyms": ["lua"]},
        {"canonical_id": "skill_perl", "display_name": "Perl", "category": "Language", "synonyms": ["perl"]},
        {"canonical_id": "skill_groovy", "display_name": "Groovy", "category": "Language", "synonyms": ["groovy"]},
        {"canonical_id": "skill_elixir", "display_name": "Elixir", "category": "Language", "synonyms": ["elixir"]},
        {"canonical_id": "skill_clojure", "display_name": "Clojure", "category": "Language", "synonyms": ["clojure"]},
        {"canonical_id": "skill_fsharp", "display_name": "F#", "category": "Language", "synonyms": ["f#", "fsharp"]},
        {"canonical_id": "skill_cobol", "display_name": "COBOL", "category": "Language", "synonyms": ["cobol"]},
        {"canonical_id": "skill_fortran", "display_name": "Fortran", "category": "Language", "synonyms": ["fortran"]},
        {"canonical_id": "skill_assembly", "display_name": "Assembly", "category": "Language", "synonyms": ["assembly", "asm"]},
        {"canonical_id": "skill_plsql", "display_name": "PL/SQL", "category": "Language", "synonyms": ["pl/sql", "plsql"]},
        {"canonical_id": "skill_tsql", "display_name": "T-SQL", "category": "Language", "synonyms": ["t-sql", "tsql"]},
        {"canonical_id": "skill_powershell", "display_name": "PowerShell", "category": "Language", "synonyms": ["powershell", "ps"]},
        {"canonical_id": "skill_ansible", "display_name": "Ansible", "category": "Tool", "synonyms": ["ansible"]},
        {"canonical_id": "skill_puppet", "display_name": "Puppet", "category": "Tool", "synonyms": ["puppet"]},
        {"canonical_id": "skill_chef", "display_name": "Chef", "category": "Tool", "synonyms": ["chef"]},
        {"canonical_id": "skill_circleci", "display_name": "CircleCI", "category": "Tool", "synonyms": ["circleci"]},
        {"canonical_id": "skill_travisci", "display_name": "Travis CI", "category": "Tool", "synonyms": ["travis ci", "travisci"]},
        {"canonical_id": "skill_github_actions", "display_name": "GitHub Actions", "category": "Tool", "synonyms": ["github actions", "gh actions"]},
        {"canonical_id": "skill_gitlab_ci", "display_name": "GitLab CI", "category": "Tool", "synonyms": ["gitlab ci", "gitlab ci/cd"]},
        {"canonical_id": "skill_bitbucket_pipelines", "display_name": "Bitbucket Pipelines", "category": "Tool", "synonyms": ["bitbucket pipelines"]},
        {"canonical_id": "skill_splunk", "display_name": "Splunk", "category": "Tool", "synonyms": ["splunk"]},
        {"canonical_id": "skill_datadog", "display_name": "Datadog", "category": "Tool", "synonyms": ["datadog"]},
        {"canonical_id": "skill_new_relic", "display_name": "New Relic", "category": "Tool", "synonyms": ["new relic", "newrelic"]},
        {"canonical_id": "skill_grafana", "display_name": "Grafana", "category": "Tool", "synonyms": ["grafana"]},
        {"canonical_id": "skill_prometheus", "display_name": "Prometheus", "category": "Tool", "synonyms": ["prometheus"]},
        {"canonical_id": "skill_maven", "display_name": "Maven", "category": "Tool", "synonyms": ["maven", "mvn"]},
        {"canonical_id": "skill_gradle", "display_name": "Gradle", "category": "Tool", "synonyms": ["gradle"]},
        {"canonical_id": "skill_webpack", "display_name": "Webpack", "category": "Tool", "synonyms": ["webpack"]},
        {"canonical_id": "skill_babel", "display_name": "Babel", "category": "Tool", "synonyms": ["babel"]},
        {"canonical_id": "skill_npm", "display_name": "npm", "category": "Tool", "synonyms": ["npm"]},
        {"canonical_id": "skill_yarn", "display_name": "Yarn", "category": "Tool", "synonyms": ["yarn"]},
    ]
    
    ontology.extend(more_skills)
    
    # ensure uniqueness
    seen = set()
    final_ontology = []
    for item in ontology:
        if item["canonical_id"] not in seen:
            seen.add(item["canonical_id"])
            final_ontology.append(item)
            
    with open('data/ontology.json', 'w', encoding='utf-8') as f:
        json.dump(final_ontology, f, indent=4)
        
    print(f"Generated data/ontology.json with {len(final_ontology)} skills.")

if __name__ == "__main__":
    build_ontology()
