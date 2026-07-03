import joblib
import pandas as pd
import numpy as np

def explain_prediction(session_data, model_path="d:/Placemux-aiml/week4-task15/src/models/proctor_model.pkl"):
    try:
        model_data = joblib.load(model_path)
    except FileNotFoundError:
        return "Model not found."
    
    model = model_data["model"]
    features = model_data["features"]
    threshold = model_data["threshold"]

    # Handle sensor faults (all signals null)
    signal_cols = ["tab_switch_count", "face_count_anomalies", "copy_paste_events"]
    if all(pd.isna(session_data.get(col)) for col in signal_cols):
        return {
            "prediction": None,
            "confidence": 0.0,
            "verdict": "no_data",
            "reason": "Sensor fault: all signals are null. Cannot evaluate session.",
            "fp_pattern": None
        }
        
    df = pd.DataFrame([session_data])
    
    # Engineer features matching model.py
    df["signal_combo_score"] = df[signal_cols].fillna(0).sum(axis=1)
    df["network_issue_derived"] = ((df["network_latency_flag"] == 1) & (df["webcam_dropout_seconds"] > 5)).astype(int)
    
    X = df[features].fillna(0)
    
    prob = model.predict_proba(X)[0][1]
    is_violation = prob >= threshold
    
    verdict = "flagged" if is_violation else "cleared"
    
    # Determine reason and FP pattern
    fp_pattern = None
    if not is_violation:
        if df["network_issue_derived"].iloc[0] == 1 and df["tab_switch_count"].iloc[0] > 3:
            fp_pattern = "network_issue"
            reason = f"Cleared: high tab-switch count ({df['tab_switch_count'].iloc[0]}) but network_issue_derived=1 — consistent with known connectivity FP pattern; model confidence {1-prob:.2f} that this is NOT a violation."
        elif df["face_count_anomalies"].iloc[0] > 0 and df["signal_combo_score"].iloc[0] == df["face_count_anomalies"].iloc[0]:
            fp_pattern = "camera_anomaly"
            reason = f"Cleared: isolated face_count_anomaly ({df['face_count_anomalies'].iloc[0]}) with no other signals. Consistent with single-camera harmless anomalies (e.g. cat walking by). Confidence {1-prob:.2f}."
        elif df["copy_paste_events"].iloc[0] > 2 and df["signal_combo_score"].iloc[0] == df["copy_paste_events"].iloc[0]:
            fp_pattern = "benign_copypaste"
            reason = f"Cleared: copy_paste_events ({df['copy_paste_events'].iloc[0]}) with no other anomalies. Likely non-code answer pasting. Confidence {1-prob:.2f}."
        elif df["signal_combo_score"].iloc[0] <= 4:
             fp_pattern = "borderline_weak_signal"
             reason = f"Cleared: signals are weak or borderline (combo score {df['signal_combo_score'].iloc[0]}). Not enough evidence of violation. Confidence {1-prob:.2f}."
        else:
            reason = f"Cleared: Model learned patterns indicate this is safe. Confidence {1-prob:.2f}."
    else:
        driving_signals = []
        if df["tab_switch_count"].iloc[0] > 3: driving_signals.append(f"tab_switch={df['tab_switch_count'].iloc[0]}")
        if df["face_count_anomalies"].iloc[0] > 0: driving_signals.append(f"face_anomalies={df['face_count_anomalies'].iloc[0]}")
        if df["copy_paste_events"].iloc[0] > 2: driving_signals.append(f"copy_paste={df['copy_paste_events'].iloc[0]}")
        
        reason = f"Flagged: Model confirms violation driven by {', '.join(driving_signals)}. Confidence {prob:.2f}."
        
    return {
        "prediction": int(is_violation),
        "confidence": float(prob) if is_violation else float(1 - prob),
        "verdict": verdict,
        "reason": reason,
        "fp_pattern": fp_pattern
    }

if __name__ == "__main__":
    test_session = {
        "tab_switch_count": 7,
        "face_count_anomalies": 0,
        "copy_paste_events": 0,
        "time_per_question_zscore": 1.2,
        "network_latency_flag": 1,
        "webcam_dropout_seconds": 15
    }
    print(explain_prediction(test_session))
