"""
Definition of Done: 'Deliberately induce the failure and confirm the
designed degradation actually happens.'

Failure scenario: the ML ranker is unavailable at serve time (crashed /
timed out / model registry unreachable). Designed degradation: the system
must NOT show an empty list and must NOT stop logging -- it must fall back
to the heuristic ranker (Stage C's FALLBACK_MODEL_NAME/VERSION) and log
that fallback identity truthfully, so nobody downstream mistakes a
heuristic result for a model result six months later (Pitfall #5 in the
study guide).

This test actually calls the serving path with the model forced to raise,
rather than asserting on the simulator's random FALLBACK_RATE draws.
"""
import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eventlog.ranked_list_logger import RankedListLogger, FALLBACK_MODEL_NAME, FALLBACK_MODEL_VERSION
from data.simulate_logs import heuristic_rank_score, build_candidate_job_features


class ModelUnavailableError(Exception):
    pass


class RankingModel:
    def __init__(self, fail: bool):
        self.fail = fail

    def score(self, df_slice):
        if self.fail:
            raise ModelUnavailableError("simulated: model registry timeout")
        return df_slice["skill_match"]  # would normally be model.predict_proba(...)


def serve_ranked_list(logger: RankedListLogger, job_slice, model: RankingModel):
    """The actual serving function under test: must always log a well-formed
    ranked list, whether the ML model succeeds or fails."""
    try:
        score = model.score(job_slice)
        model_name, model_version = "ltr_ranker", "mv-2026.07.15"
    except ModelUnavailableError:
        score = heuristic_rank_score(job_slice)
        model_name, model_version = FALLBACK_MODEL_NAME, FALLBACK_MODEL_VERSION

    ranked = job_slice.assign(score=score).sort_values("score", ascending=False).head(10)
    impressions = logger.log_ranked_list(str(uuid.uuid4()), str(uuid.uuid4()),
                                          ranked["candidate_id"].tolist(), model_name, model_version)
    return impressions, model_name, model_version


def run():
    pairs = build_candidate_job_features()
    job_slice = pairs[pairs.job_id == "job_0"].head(20)
    test_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "artifacts", "failure_test_log.csv")
    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    logger = RankedListLogger(test_log_path)

    # 1. Healthy path
    healthy_impressions, hn, hv = serve_ranked_list(logger, job_slice, RankingModel(fail=False))
    assert hn == "ltr_ranker" and len(healthy_impressions) == 10

    # 2. Induced failure path
    failed_impressions, fn, fv = serve_ranked_list(logger, job_slice, RankingModel(fail=True))
    assert fn == FALLBACK_MODEL_NAME and fv == FALLBACK_MODEL_VERSION, "did not degrade to fallback ranker"
    assert len(failed_impressions) == 10, "fallback path served an incomplete/empty list"
    assert all(e.model_name == FALLBACK_MODEL_NAME for e in failed_impressions), "fallback identity not logged truthfully"

    print("PASS: healthy path used ltr_ranker/mv-2026.07.15")
    print(f"PASS: induced-failure path degraded to {fn}/{fv} and still served+logged {len(failed_impressions)} items")
    print(f"Evidence log written to {test_log_path}")
    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
