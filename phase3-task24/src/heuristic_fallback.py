"""
Deterministic, model-free heuristic ranker. This is the 'sane heuristic' the
study guide requires matching to degrade to when the ML model dies -- it must
never throw, never depend on the model service or feature freshness beyond
raw inputs, and always return a usable score.
"""


class HeuristicRanker:
    WEIGHTS = {"skill_match": 0.5, "exp_match": 0.3, "location_match": 0.2}

    def score(self, feats: dict) -> float:
        skill = feats.get("skill_match")
        exp = feats.get("exp_match")
        loc = feats.get("location_match")

        # defend against corrupted/NaN inputs even in the fallback path
        skill = skill if (skill == skill and 0 <= skill <= 1) else 0.3
        exp = exp if (exp == exp and 0 <= exp <= 1) else 0.3
        loc = loc if loc in (0, 1) else 0

        return (self.WEIGHTS["skill_match"] * skill +
                self.WEIGHTS["exp_match"] * exp +
                self.WEIGHTS["location_match"] * loc)
