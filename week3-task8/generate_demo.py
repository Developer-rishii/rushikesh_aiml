import json
from src.baseline_matcher import BaselineMatcher
from src.guardrail import evaluate_guardrail

matcher = BaselineMatcher('d:/Placemux-aiml/week3-task8/data/candidate_profiles.csv', 'd:/Placemux-aiml/week3-task8/data/jobs.csv')
with open('d:/Placemux-aiml/week3-task8/metrics/guardrail_metrics.json', 'r') as f:
    metrics = json.load(f)
    threshold = metrics['threshold_used']

# Find a good match (high score)
good_res = None
bad_res = None
for i in range(1, 100):
    for j in range(1, 10):
        match_data = matcher.match(i, j)
        res = evaluate_guardrail(match_data, threshold)
        if res['fit_status'] == 'OK' and not good_res:
            good_res = res
        elif res['fit_status'] == 'LOW_FIT_WARNING' and not bad_res:
            bad_res = res
        if good_res and bad_res: break
    if good_res and bad_res: break

print("---GOOD---")
print(json.dumps(good_res, indent=2))
print("---BAD---")
print(json.dumps(bad_res, indent=2))
