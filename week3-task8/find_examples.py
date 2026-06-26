import pandas as pd
df = pd.read_csv('d:/Placemux-aiml/week3-task8/data/match_history.csv')

# Good match example (Score > 40)
good = df[df['prediction_score'] > 60].iloc[0]
print(f"GOOD: {int(good['candidate_id'])}, {int(good['job_id'])}")

# Bad match example (Score < 40)
bad = df[df['prediction_score'] < 30].iloc[0]
print(f"BAD: {int(bad['candidate_id'])}, {int(bad['job_id'])}")

# Borderline example (~40)
border = df[(df['prediction_score'] >= 38) & (df['prediction_score'] <= 42)].iloc[0]
print(f"BORDERLINE: {int(border['candidate_id'])}, {int(border['job_id'])}")
