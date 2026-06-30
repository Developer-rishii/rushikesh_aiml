"""
Full Parsing v0 Pipeline: Stage 3a (rule-based candidate generation) + 
Stage 3b (trained ML model filtering).

For each extracted skill, emits a plain-English explanation combining both stages.
"""
import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.ontology import SkillsOntology
from src.rule_extractor import extract_candidates
from src.model_trainer import featurize, FEATURE_NAMES
from src.baseline import extract_baseline


class ParsingPipeline:
    def __init__(self, ontology_path='data/skills_ontology.csv', 
                 model_path='src/models/skill_classifier.pkl',
                 confidence_threshold=0.5):
        self.ontology = SkillsOntology(ontology_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run model_trainer.py first.")
        self.model = joblib.load(model_path)
        self.confidence_threshold = confidence_threshold
        self.feature_names = FEATURE_NAMES
    
    def parse(self, text, ground_truth=None):
        """
        Parse text and return structured skill extraction results.
        
        Returns:
            dict with keys:
                - status: 'ok' or 'no_skills_found'
                - skills: list of extracted skill dicts
                - misses: list of ground-truth skills not extracted (if gt provided)
        """
        if not text or not str(text).strip():
            return {
                'status': 'no_skills_found',
                'skills': [],
                'misses': list(ground_truth) if ground_truth else [],
                'explanation': 'Input text is empty or contains only whitespace.'
            }
        
        # Stage 3a: Rule-based candidate generation
        candidates = extract_candidates(text, self.ontology)
        
        if not candidates:
            return {
                'status': 'no_skills_found',
                'skills': [],
                'misses': list(ground_truth) if ground_truth else [],
                'explanation': 'No skill candidates found in the text.'
            }
        
        # Stage 3b: ML model filtering
        extracted_skills = []
        for cand in candidates:
            features = featurize(cand)
            X = np.array([[features[f] for f in self.feature_names]])
            
            proba = self.model.predict_proba(X)[0]
            confidence = proba[1] if len(proba) > 1 else proba[0]
            
            # Get feature importances for explanation
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Determine top contributing features for THIS prediction
            feature_values = {f: features[f] for f in self.feature_names}
            top_drivers = []
            for feat_name, feat_imp in top_features:
                val = feature_values[feat_name]
                if feat_name == 'is_negated':
                    driver_str = f"{'negation detected' if val else 'no nearby negation cue'}"
                elif feat_name == 'fuzzy_score':
                    driver_str = f"fuzzy_score={val:.2f}"
                elif feat_name.startswith('match_'):
                    mtype = feat_name.replace('match_', '')
                    driver_str = f"match_type={'is' if val else 'not'} {mtype}"
                elif feat_name == 'is_short_token':
                    driver_str = f"{'short/ambiguous token' if val else 'normal-length token'}"
                else:
                    driver_str = f"{feat_name}={val}"
                top_drivers.append(driver_str)
            
            skill_result = {
                'canonical_name': cand['canonical_name'],
                'match_type': cand['match_type'],
                'matched_text': cand['matched_text'],
                'offset_start': cand['offset_start'],
                'offset_end': cand['offset_end'],
                'fuzzy_score': cand['fuzzy_score'],
                'confidence': round(float(confidence), 4),
                'is_negated': cand['is_negated'],
                'context': cand['context'],
                'explanation': (
                    f"Extracted '{cand['canonical_name']}': matched "
                    f"{'canonical name' if cand['match_type'] == 'exact' else cand['match_type']} "
                    f"'{cand['matched_text']}' at offset {cand['offset_start']}; "
                    f"model confidence {confidence:.2f}, driven mainly by "
                    f"{' and '.join(top_drivers)}"
                )
            }
            
            if confidence >= self.confidence_threshold and not cand['is_negated']:
                extracted_skills.append(skill_result)
        
        # Compute misses
        misses = []
        if ground_truth:
            extracted_names = {s['canonical_name'] for s in extracted_skills}
            for gt_skill in ground_truth:
                if gt_skill not in extracted_names:
                    # Find if it was a candidate that got filtered
                    was_candidate = any(c['canonical_name'] == gt_skill for c in candidates)
                    filtered_cand = next((c for c in candidates if c['canonical_name'] == gt_skill), None)
                    
                    miss_info = {
                        'canonical_name': gt_skill,
                        'reason': 'not_found_in_text' if not was_candidate else (
                            'filtered_by_negation' if filtered_cand and filtered_cand['is_negated'] else
                            'filtered_by_model_low_confidence'
                        )
                    }
                    if filtered_cand:
                        features = featurize(filtered_cand)
                        X = np.array([[features[f] for f in self.feature_names]])
                        proba = self.model.predict_proba(X)[0]
                        conf = proba[1] if len(proba) > 1 else proba[0]
                        miss_info['candidate_confidence'] = round(float(conf), 4)
                        miss_info['matched_text'] = filtered_cand['matched_text']
                        miss_info['match_type'] = filtered_cand['match_type']
                    
                    misses.append(miss_info)
        
        return {
            'status': 'ok' if extracted_skills else 'no_skills_found',
            'skills': extracted_skills,
            'misses': misses,
        }
    
    def parse_baseline(self, text):
        """Run baseline extraction for comparison."""
        return extract_baseline(text, self.ontology)
    
    def parse_rules_only(self, text, ground_truth=None):
        """Run Stage 3a only (no ML filtering) — for comparison."""
        if not text or not str(text).strip():
            return {
                'status': 'no_skills_found',
                'skills': [],
                'misses': list(ground_truth) if ground_truth else [],
            }
        
        candidates = extract_candidates(text, self.ontology)
        # Return all non-negated candidates as "extracted"
        extracted = [
            {
                'canonical_name': c['canonical_name'],
                'match_type': c['match_type'],
                'matched_text': c['matched_text'],
                'is_negated': c['is_negated'],
            }
            for c in candidates if not c['is_negated']
        ]
        
        misses = []
        if ground_truth:
            extracted_names = {s['canonical_name'] for s in extracted}
            for gt in ground_truth:
                if gt not in extracted_names:
                    misses.append({'canonical_name': gt, 'reason': 'not_found'})
        
        return {
            'status': 'ok' if extracted else 'no_skills_found',
            'skills': extracted,
            'misses': misses,
        }
