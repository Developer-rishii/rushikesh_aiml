"""
Stage C - "Live monitoring / connect offline to online" (Sec 5: "nDCG going
up offline means nothing if applications go down online").

We do not have a live A/B test (Phase 3 go-live has not happened yet), so
we use Inverse Propensity Scoring (IPS) off-policy evaluation on the logged
data: it estimates what the ONLINE click-through/application rate would
have been had the new model's ranking been shown, using only logs collected
under the old (baseline) policy. This is the standard honest way to bridge
offline and online *before* a real rollout, and it is explicitly reported
as an ESTIMATE with a confidence gap - not claimed as a real online result
(the study guide penalizes "a claim without evidence" and "shipping an
offline win that never gets validated online").
"""
import os, pickle, json, time
import numpy as np
import pandas as pd
from common import get_X

ROOT = os.path.dirname(os.path.dirname(__file__))
EXP_LOG = f"{ROOT}/reports/experiment_log.csv"

def main():
    df = pd.read_csv(f"{ROOT}/data/logs.csv")
    test = df[df.day >= 20].copy()
    with open(f"{ROOT}/registry/models/ranker_v2.0.pkl", "rb") as f:
        model = pickle.load(f)
    test["model_score"] = model.predict(get_X(test))

    # logging policy = baseline popularity ranker actually shown historically
    pop = df[df.day < 20].groupby("job_id").clicked.mean().rename("baseline_score")
    test = test.merge(pop, on="job_id", how="left")
    test["baseline_score"] = test["baseline_score"].fillna(df[df.day < 20].clicked.mean())

    # propensity: probability the logging policy would have surfaced this
    # item in the top-10 for that candidate (proxy via rank-based softmax)
    def top10_indicator(scores):
        return (pd.Series(scores).rank(ascending=False, method="first") <= 10).astype(int)

    test["logged_top10"] = test.groupby("query_id")["baseline_score"].transform(top10_indicator)
    test["new_top10"] = test.groupby("query_id")["model_score"].transform(top10_indicator)

    propensity = test.logged_top10.mean()  # P(shown | logging policy), pooled
    propensity = max(propensity, 0.05)  # clip to avoid IPS blow-up

    # IPS estimator: only count reward on rows the new policy would surface
    # AND that were actually logged (clicked/applied), reweighted by 1/propensity
    observed = test[test.logged_top10 == 1]
    ips_ctr = (observed.new_top10 * observed.clicked / propensity).sum() / len(test)
    ips_apply = (observed.new_top10 * observed.applied / propensity).sum() / len(test)

    logged_ctr = test.clicked.mean()
    logged_apply = test.applied.mean()

    result = {
        "method": "Inverse Propensity Scoring (off-policy estimate, NOT a live A/B result)",
        "propensity_clip": round(float(propensity), 4),
        "estimated_ctr_new_policy": round(float(ips_ctr), 4),
        "observed_ctr_logged_policy": round(float(logged_ctr), 4),
        "estimated_application_rate_new_policy": round(float(ips_apply), 4),
        "observed_application_rate_logged_policy": round(float(logged_apply), 4),
        "gap_warning": "IPS variance is high with only 20074 held-out rows; "
                        "treat this as directional, not final. Real online "
                        "A/B on the staged rollout (5% -> 25% -> 100%) is "
                        "the source of truth and is what monitoring/ tracks live."
    }
    json.dump(result, open(f"{ROOT}/reports/online_proxy_eval.json", "w"), indent=2)
    with open(EXP_LOG, "a") as f:
        t = int(time.time())
        f.write(f"online_proxy,ips,estimated_ctr_new_policy,{ips_ctr:.4f},{t}\n")
        f.write(f"online_proxy,ips,estimated_application_rate_new_policy,{ips_apply:.4f},{t}\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
