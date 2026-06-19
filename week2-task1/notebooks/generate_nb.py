import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell('# PlaceMux Matching Engine: End-to-End Demo\n\nThis notebook demonstrates the end-to-end flow of the matching engine: Data loading -> Feature Engineering -> Matching -> Ranking -> Explainability.'),
    nbf.v4.new_code_cell('import os, sys\nimport pandas as pd\nsys.path.append(os.path.abspath(".."))\nfrom src.feature_engineering import get_feature_spaces\nfrom src.matcher import calculate_match\nfrom src.explainability import generate_reasons\nfrom src.ranking import rank_candidates\nfrom src.metrics import simulate_ground_truth_and_evaluate'),
    nbf.v4.new_markdown_cell('## 1. Data Loading and Feature Engineering'),
    nbf.v4.new_code_cell('students_df, jobs_df = get_feature_spaces("../data/students.csv", "../data/jobs.csv")\ndisplay(students_df.head())\ndisplay(jobs_df.head())'),
    nbf.v4.new_markdown_cell('## 2. Match Single Student to Single Job'),
    nbf.v4.new_code_cell('student = students_df.iloc[0].to_dict()\njob = jobs_df.iloc[0].to_dict()\nprint(f\'Student: {student["Name"]}\')\nprint(f\'Job: {job["Role"]} at {job["Company Name"]}\')\n\nscore, details = calculate_match(student, job)\nreasons = generate_reasons(score, details)\n\nprint(f"\\nMatch Score: {score}%")\nprint("\\nReasons:")\nfor r in reasons:\n    print(r)'),
    nbf.v4.new_markdown_cell('## 3. Top-N Ranking'),
    nbf.v4.new_code_cell('job = jobs_df.iloc[0].to_dict()\nprint(f\'Ranking for Job: {job["Role"]} at {job["Company Name"]}\')\n\nranked = rank_candidates(job, students_df, top_n=3)\nfor rank, candidate in enumerate(ranked, 1):\n    print(f\'\\nRank {rank}: {candidate["student_name"]} (Score: {candidate["match_score"]}%)\')\n    for r in candidate["reasons"]:\n        print(f"  {r}")'),
    nbf.v4.new_markdown_cell('## 4. Evaluation Metrics'),
    nbf.v4.new_code_cell('metrics = simulate_ground_truth_and_evaluate(calculate_match, students_df, jobs_df)\nprint("\\nEvaluation Metrics:")\nfor k, v in metrics.items():\n    print(f"{k}: {v:.4f}")')
]
nb['cells'] = cells
nbf.write(nb, 'exploration.ipynb')
print("Notebook generated successfully.")
