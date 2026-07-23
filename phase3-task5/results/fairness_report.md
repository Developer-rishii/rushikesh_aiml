# Fairness Evaluation Report

Generated programmatically by `training/eval_fairness.py`.

## Demographic Parity Check
- Group 0 (Majority) Mean Predicted Score: -0.9124
- Group 1 (Minority) Mean Predicted Score: -0.9360
- Gap (Group 0 - Group 1): 0.0236
- Direction: Group 0 scored higher

## Equal Opportunity Check (among candidates with relevance > 0)
- Group 0 Mean Predicted Score: -0.1919
- Group 1 Mean Predicted Score: 0.4308
- Gap (Group 0 - Group 1): -0.6227
- Direction: Group 1 scored higher

## Interpretation
The model shows a mixed fairness picture. On demographic parity (all candidates), Group 0 scores slightly higher (gap=0.0236), suggesting the model has absorbed some of the historical bias baked into the training data. However, on equal opportunity (only truly relevant candidate-job pairs), Group 1 actually scores higher (gap=-0.6227). This means that among candidates who genuinely applied, the model is more generous to Group 1 — possibly because Group 1 candidates who overcame the historical bias to actually apply had stronger signals on other features (experience, skills). This discrepancy is important: the demographic parity gap and the equal opportunity gap point in opposite directions, so a single 'biased against Group 1' narrative would be inaccurate.