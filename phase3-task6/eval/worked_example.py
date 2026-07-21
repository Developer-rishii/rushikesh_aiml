"""
'Make it explainable, safe & demoable: produce one worked example: this
input, this output, this plain-English reason - plus what happens when
the model is unavailable.'
"""
import joblib
import pandas as pd
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.simulate_logs import FEATURE_COLS

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def run():
    model = joblib.load(os.path.join(ARTIFACTS, "ranker_model.joblib"))
    test_df = pd.read_csv(os.path.join(ARTIFACTS, "test_predictions.csv"))
    example = test_df.sort_values("pred_score", ascending=False).iloc[0]

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.round(3).tolist()))
    top_feature = max(importances, key=importances.get)

    result = {
        "input": {c: round(float(example[c]), 3) for c in FEATURE_COLS},
        "output": {
            "predicted_click_probability": round(float(example["pred_score"]), 3),
            "position_it_was_shown_at": int(example["position"]),
            "model_version": example["model_version"],
        },
        "plain_english_reason": (
            f"This candidate-job pair scored highly mainly because of '{top_feature}' "
            f"(the single most influential feature learned by the model, importance="
            f"{importances[top_feature]}), combined with the other logged features. "
            f"It was shown at position {int(example['position'])} by model "
            f"{example['model_version']}, and that exact model identity was stamped on "
            f"the impression event so this explanation stays valid even if the model "
            f"is retrained tomorrow."
        ),
        "what_happens_if_model_is_unavailable": (
            "Serving falls back to the heuristic_fallback_ranker (see "
            "eval/failure_injection_test.py) instead of failing to serve a list. "
            "The fallback's identity is logged truthfully (model_name="
            "'heuristic_fallback_ranker'), so this worked example's explanation would "
            "never be silently attributed to a fallback decision, or vice versa."
        ),
        "feature_importances": importances,
    }
    with open(os.path.join(ARTIFACTS, "worked_example.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    run()
