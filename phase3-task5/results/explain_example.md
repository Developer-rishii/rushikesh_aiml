# Explainability Report

Generated programmatically by `training/explain.py`.
Base Value (Average Prediction across all data): -0.9431

---
## Example 1: Candidate C107 -> Job J10
- **Predicted Score:** -0.0874

### Feature Contributions
| Feature | Value | SHAP Contribution |
|---------|-------|--------------------|
| candidate_exp | 7.00 | +0.3210 |
| candidate_skills | 7.00 | +0.3227 |
| required_exp | 4.00 | +0.2830 |
| required_skills | 4.00 | +0.6048 |
| job_popularity | 0.29 | -0.6758 |

### Plain English Explanation
The model predicted a match score of -0.0874 for candidate C107 and job J10. The biggest reason for increasing this score was 'required_skills' (value: 4.00), which pushed the score up by 0.6048. On the other hand, the score was brought down mostly by 'job_popularity' (value: 0.29), which reduced the score by 0.6758.

---
## Example 2: Candidate C107 -> Job J106
- **Predicted Score:** -1.1707

### Feature Contributions
| Feature | Value | SHAP Contribution |
|---------|-------|--------------------|
| candidate_exp | 7.00 | +0.2129 |
| candidate_skills | 7.00 | +0.4328 |
| required_exp | 7.00 | -0.4262 |
| required_skills | 7.00 | -0.3661 |
| job_popularity | 0.56 | -0.0810 |

### Plain English Explanation
The model predicted a match score of -1.1707 for candidate C107 and job J106. The biggest reason for increasing this score was 'candidate_skills' (value: 7.00), which pushed the score up by 0.4328. On the other hand, the score was brought down mostly by 'required_exp' (value: 7.00), which reduced the score by 0.4262.

---
## Example 3: Candidate C107 -> Job J146
- **Predicted Score:** -0.1541

### Feature Contributions
| Feature | Value | SHAP Contribution |
|---------|-------|--------------------|
| candidate_exp | 7.00 | -0.1844 |
| candidate_skills | 7.00 | +0.5594 |
| required_exp | 0.00 | +0.6372 |
| required_skills | 5.00 | +0.1781 |
| job_popularity | 0.39 | -0.4013 |

### Plain English Explanation
The model predicted a match score of -0.1541 for candidate C107 and job J146. The biggest reason for increasing this score was 'required_exp' (value: 0.00), which pushed the score up by 0.6372. On the other hand, the score was brought down mostly by 'job_popularity' (value: 0.39), which reduced the score by 0.4013.

---
## Example 4: Candidate C107 -> Job J250
- **Predicted Score:** 1.9259

### Feature Contributions
| Feature | Value | SHAP Contribution |
|---------|-------|--------------------|
| candidate_exp | 7.00 | +0.2696 |
| candidate_skills | 7.00 | +0.4663 |
| required_exp | 3.00 | +0.5537 |
| required_skills | 4.00 | +0.6444 |
| job_popularity | 0.90 | +0.9350 |

### Plain English Explanation
The model predicted a match score of 1.9259 for candidate C107 and job J250. The biggest reason for increasing this score was 'job_popularity' (value: 0.90), which pushed the score up by 0.9350.

---
## Example 5: Candidate C107 -> Job J263
- **Predicted Score:** -3.1545

### Feature Contributions
| Feature | Value | SHAP Contribution |
|---------|-------|--------------------|
| candidate_exp | 7.00 | -0.1363 |
| candidate_skills | 7.00 | -0.2840 |
| required_exp | 0.00 | +0.4234 |
| required_skills | 9.00 | -1.9813 |
| job_popularity | 0.40 | -0.2331 |

### Plain English Explanation
The model predicted a match score of -3.1545 for candidate C107 and job J263. The biggest reason for increasing this score was 'required_exp' (value: 0.00), which pushed the score up by 0.4234. On the other hand, the score was brought down mostly by 'required_skills' (value: 9.00), which reduced the score by 1.9813.
