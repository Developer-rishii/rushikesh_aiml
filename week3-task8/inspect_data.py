import pandas as pd
candidates = pd.read_csv('d:/Placemux-aiml/week3-task8/data/candidate_profiles.csv')
jobs = pd.read_csv('d:/Placemux-aiml/week3-task8/data/jobs.csv')
print("Candidates columns:", list(candidates.columns))
print("Jobs columns:", list(jobs.columns))
