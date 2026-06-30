"""
Evaluator: computes precision, recall, FPR for baseline, rule-only, and full pipeline
across all sample data. Segments by document type and hard cases.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.pipeline import ParsingPipeline


def compute_metrics(extracted_names, ground_truth_names):
    """Compute precision, recall, false positive rate for a single document."""
    extracted = set(extracted_names)
    gt = set(ground_truth_names)
    
    tp = len(extracted & gt)
    fp = len(extracted - gt)
    fn = len(gt - extracted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # no extractions = vacuous precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # no gt skills = vacuous recall
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'extracted': list(extracted),
        'ground_truth': list(gt),
        'false_positives': list(extracted - gt),
        'misses': list(gt - extracted),
    }


def aggregate_metrics(doc_results):
    """Aggregate per-document metrics into overall precision/recall/FPR."""
    total_tp = sum(r['tp'] for r in doc_results)
    total_fp = sum(r['fp'] for r in doc_results)
    total_fn = sum(r['fn'] for r in doc_results)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    fpr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    return {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'fpr': round(fpr, 4),
        'num_docs': len(doc_results),
    }


# IDs for hard-case documents
HARD_CASE_IDS = {
    'junk': ['res_junk_01'],
    'alias_only': ['res_alias_01'],
    'negation': ['res_negation_01'],
    'empty': ['jd_empty_01'],
    'substring_trap': ['res_substring_01'],
}


def run_evaluation(pipeline):
    """Run full evaluation across all sample data."""
    results = {
        'baseline': {'resume': [], 'jd': [], 'hard_cases': {}},
        'rules_only': {'resume': [], 'jd': [], 'hard_cases': {}},
        'full_pipeline': {'resume': [], 'jd': [], 'hard_cases': {}},
    }
    
    all_hard_ids = set()
    for ids in HARD_CASE_IDS.values():
        all_hard_ids.update(ids)
    
    for data_file, doc_type in [('data/resumes.json', 'resume'), ('data/jds.json', 'jd')]:
        with open(data_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        for doc in docs:
            doc_id = doc['doc_id']
            text = doc['text']
            gt_skills = doc['ground_truth']
            
            # Baseline
            baseline_extracted = pipeline.parse_baseline(text)
            baseline_metrics = compute_metrics(baseline_extracted, gt_skills)
            baseline_metrics['doc_id'] = doc_id
            results['baseline'][doc_type].append(baseline_metrics)
            
            # Rules only (Stage 3a, no ML filtering)
            rules_result = pipeline.parse_rules_only(text, gt_skills)
            rules_extracted = [s['canonical_name'] for s in rules_result['skills']]
            rules_metrics = compute_metrics(rules_extracted, gt_skills)
            rules_metrics['doc_id'] = doc_id
            results['rules_only'][doc_type].append(rules_metrics)
            
            # Full pipeline (Stage 3a + 3b)
            full_result = pipeline.parse(text, gt_skills)
            full_extracted = [s['canonical_name'] for s in full_result['skills']]
            full_metrics = compute_metrics(full_extracted, gt_skills)
            full_metrics['doc_id'] = doc_id
            results['full_pipeline'][doc_type].append(full_metrics)
            
            # Track hard cases
            for case_name, case_ids in HARD_CASE_IDS.items():
                if doc_id in case_ids:
                    for method in ['baseline', 'rules_only', 'full_pipeline']:
                        if case_name not in results[method]['hard_cases']:
                            results[method]['hard_cases'][case_name] = []
                        # Find the metrics we just computed
                        if method == 'baseline':
                            results[method]['hard_cases'][case_name].append(baseline_metrics)
                        elif method == 'rules_only':
                            results[method]['hard_cases'][case_name].append(rules_metrics)
                        else:
                            results[method]['hard_cases'][case_name].append(full_metrics)
    
    # Aggregate
    report = {}
    for method in ['baseline', 'rules_only', 'full_pipeline']:
        report[method] = {
            'resume': aggregate_metrics(results[method]['resume']),
            'jd': aggregate_metrics(results[method]['jd']),
            'overall': aggregate_metrics(results[method]['resume'] + results[method]['jd']),
            'hard_cases': {}
        }
        for case_name in HARD_CASE_IDS:
            case_docs = results[method]['hard_cases'].get(case_name, [])
            if case_docs:
                agg = aggregate_metrics(case_docs)
                # Also include per-doc details for the report
                agg['details'] = case_docs
                report[method]['hard_cases'][case_name] = agg
    
    return report


def save_report(report, output_path='reports/evaluation_results.json'):
    """Save the evaluation report to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Evaluation report saved to {output_path}")
    return report


if __name__ == '__main__':
    pipeline = ParsingPipeline()
    report = run_evaluation(pipeline)
    save_report(report)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    for method in ['baseline', 'rules_only', 'full_pipeline']:
        m = report[method]['overall']
        print(f"\n{method.upper()}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  FPR:       {m['fpr']:.4f}")
        print(f"  (TP={m['total_tp']}, FP={m['total_fp']}, FN={m['total_fn']})")
