"""
run_pipeline.py
Stage E.1: "Run the full end-to-end journey on real [logged] data."
Orchestrates: data generation -> stuffing detector train/eval -> ranker
train/eval -> extraction detector -> poison detector -> writes a single
versioned experiment_log.json (Threat T7: model versioning / accountability).
"""
import json, os, sys, hashlib, datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "ranking_defense"))
sys.path.insert(0, os.path.join(ROOT, "extraction_poison_detection"))

import generate_data
from stuffing_detector import train_and_eval as train_stuffing
from ranker import train_ranker, robust_features
from stuffing_detector import rule_signals
from evaluate import evaluate as evaluate_ranker
from extraction_detector import detect as detect_extraction
from poison_detector import inject_poison, detect as detect_poison


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    log = {"run_timestamp_utc": datetime.datetime.utcnow().isoformat(), "seed": 42, "stages": {}}

    # Stage: data
    generate_data.main()
    log["stages"]["data_generation"] = "OK -- see data/*.json"

    with open(os.path.join(ROOT, "data", "candidates.json")) as f:
        candidates = json.load(f)
    with open(os.path.join(ROOT, "data", "jobs.json")) as f:
        jobs = json.load(f)
    with open(os.path.join(ROOT, "data", "interactions.json")) as f:
        interactions = json.load(f)

    # Stage C: stuffing detector
    stuffing_result, clf = train_stuffing(candidates, os.path.join(ROOT, "ranking_defense"))
    log["stages"]["stuffing_detector_eval"] = stuffing_result

    stuffing_scores = {c["candidate_id"]: rule_signals(c["resume_text"])["repetition_rate"]
                        for c in candidates}

    # Stage C: robust ranker
    model, test_interactions = train_ranker(candidates, jobs, interactions, stuffing_scores,
                          os.path.join(ROOT, "ranking_defense"))
    ranking_result = evaluate_ranker(candidates, jobs, test_interactions, model, stuffing_scores)
    log["stages"]["ranking_eval"] = ranking_result

    # Stage D: extraction detection
    extraction_result, _ = detect_extraction(interactions, os.path.join(ROOT, "extraction_poison_detection"))
    log["stages"]["extraction_detection_eval"] = extraction_result

    # Stage D: poison detection
    poisoned, n_poison = inject_poison(interactions, candidates, jobs)
    poison_result, _ = detect_poison(poisoned, candidates, jobs, os.path.join(ROOT, "extraction_poison_detection"))
    log["stages"]["poison_detection_eval"] = poison_result

    # Model versioning (Threat T7)
    ranker_path = os.path.join(ROOT, "ranking_defense", "ranker.joblib")
    clf_path = os.path.join(ROOT, "ranking_defense", "stuffing_clf.joblib")
    log["model_versions"] = {
        "ranker_model_hash": file_hash(ranker_path),
        "stuffing_classifier_hash": file_hash(clf_path),
    }

    # Scoring self-check against rubric before writing the log
    log["definition_of_done_self_check"] = {
        "threat_model_delivered": os.path.exists(os.path.join(ROOT, "threat_model", "threat_model.md")),
        "stuffing_defence_beats_baseline": (
            stuffing_result["trained_classifier"]["f1"] >= stuffing_result["baseline_rule_only"]["f1"]
        ),
        "ranker_beats_naive_on_nDCG": (
            ranking_result["offline_nDCG_at_k"]["robust_model"]
            >= ranking_result["offline_nDCG_at_k"]["naive_stuffable_baseline"]
        ),
        "extraction_detection_recall_nonzero": extraction_result["recall"] > 0,
        "poison_detection_recall_nonzero": poison_result["recall"] > 0,
        "fairness_gap_reported_every_run": ranking_result["fairness_max_group_gap"] is not None,
    }

    out_path = os.path.join(ROOT, "experiment_log.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(log["definition_of_done_self_check"], indent=2))
    return log


if __name__ == "__main__":
    main()
