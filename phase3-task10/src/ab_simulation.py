"""
ab_simulation.py
-----------------
Runs AFTER pre_registration.json is locked (see run_all.sh ordering, and the
_enforce_no_peeking guard in readout.py which checks file timestamps).

Feeds the trained baseline and treatment scoring functions into the online
event simulator from data_simulation.py for the pre-registered 14-day
window, and writes data/online_ab_events.csv — the only input readout.py
is allowed to read.
"""

import pandas as pd

from data_simulation import simulate_ab_events
from train_ranker import baseline_score_fn, train


def main():
    hist = pd.read_csv("data/historical_logs.csv")
    treatment, _, _ = train(hist)

    events = simulate_ab_events(baseline_score_fn, treatment.score)
    events.to_csv("data/online_ab_events.csv", index=False)
    print(f"wrote data/online_ab_events.csv rows={len(events)} "
          f"queries={events.query_id.nunique()} "
          f"arms={events.arm.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
