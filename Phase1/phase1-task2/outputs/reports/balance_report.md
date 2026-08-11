# Class Balance Report

| class (0=malignant, 1=benign) | count | rate |
|---|---|---|
| 0 | 212 | 37.26% |
| 1 | 357 | 62.74% |

**Majority-class baseline accuracy: 62.74%**

The classes are moderately imbalanced (~63/37), not extreme, but a majority-class baseline already scores 62.7% accuracy, so accuracy alone is a weak signal of real skill here. PR-AUC (see config.py SUCCESS_METRIC) is tracked alongside it.

Imbalance ratio (majority:minority): 1.68:1
Ratio is mild -> stratified split is sufficient, class weighting optional (still applied for margin of safety).