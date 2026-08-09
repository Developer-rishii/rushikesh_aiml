"""
Stage C - "Live monitoring through the v2.0 rollout".
Replays the held-out days as a staged rollout (5% -> 25% -> 100% traffic,
Sec 8 alternative approaches) and computes Population Stability Index (PSI)
per feature per day against the training distribution. This is what
catches the deliberately-injected serving bug in generate_data.py (exp_years
under-logged from day 20) - i.e. the exact train/serve skew failure mode
named in Sec 5 as "the single biggest silent killer".

Defines and answers the brainstorming question: "What is the ONE number
that tells you to roll back?" -> answer: PSI(exp_years) > 0.25 (Sec 9).
"""
import os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
EXP_LOG = f"{ROOT}/reports/experiment_log.csv"
ROLLBACK_FEATURE = "exp_years"
ROLLBACK_PSI_THRESHOLD = 0.25  # industry rule of thumb: >0.25 = major shift, act now

def psi(expected, actual, bins=10):
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, breakpoints)[0] / len(expected)
    a_perc = np.histogram(actual, breakpoints)[0] / len(actual)
    e_perc = np.clip(e_perc, 1e-4, None)
    a_perc = np.clip(a_perc, 1e-4, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def main():
    df = pd.read_csv(f"{ROOT}/data/logs.csv")
    reference = df[df.day < 20]
    features = ["skill_score", "exp_years", "job_seniority", "job_comp_level"]

    rows = []
    rollback_day = None
    for day in sorted(df.day.unique()):
        if day < 20:
            continue
        day_df = df[df.day == day]
        # staged rollout traffic ramp for illustration
        stage = "5%" if day < 22 else ("25%" if day < 25 else "100%")
        day_psi = {feat: psi(reference[feat].values, day_df[feat].values) for feat in features}
        breach = day_psi[ROLLBACK_FEATURE] > ROLLBACK_PSI_THRESHOLD
        if breach and rollback_day is None:
            rollback_day = int(day)
        rows.append({"day": int(day), "rollout_stage": stage, **{f"psi_{k}": round(v, 4) for k, v in day_psi.items()},
                     "rollback_triggered": bool(breach)})

    monitor_df = pd.DataFrame(rows)
    monitor_df.to_csv(f"{ROOT}/reports/rollout_monitor_log.csv", index=False)

    result = {
        "rollback_trigger_metric": f"PSI({ROLLBACK_FEATURE}) vs pre-rollout training distribution",
        "rollback_threshold": ROLLBACK_PSI_THRESHOLD,
        "rollback_triggered": rollback_day is not None,
        "rollback_triggered_on_day": rollback_day,
        "action_on_trigger": "freeze traffic ramp at current stage, page on-call, "
                               "revert serving to previous model version within 5 min",
        "root_cause_found": f"feature '{ROLLBACK_FEATURE}' under-logged by serving "
                              "pipeline starting day 20 (simulated real bug)",
    }
    json.dump(result, open(f"{ROOT}/reports/rollback_decision.json", "w"), indent=2)
    with open(EXP_LOG, "a") as f:
        t = int(time.time())
        f.write(f"monitoring,rollout,rollback_triggered,{result['rollback_triggered']},{t}\n")
        f.write(f"monitoring,rollout,rollback_day,{rollback_day},{t}\n")
    print(monitor_df.to_string(index=False))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
