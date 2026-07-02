"""
Layered Skill Mapper — maps raw parsed skill strings to canonical ontology nodes.

Architecture (executed in order, first confident match wins):
  Layer 1: Exact / synonym-table lookup         → confidence 1.0
  Layer 2: Fuzzy string match (edit distance + token overlap) → confidence 0.5–0.95
  Layer 3: TF-IDF cosine similarity              → confidence 0.4–0.9

Below a tuned threshold, returns "unmapped" rather than guessing.
"""

import json
import os
import re
import csv
import datetime
from difflib import SequenceMatcher
from collections import defaultdict


class SkillMapper:
    """Three-layer skill mapper with confidence scoring and experiment logging."""

    # Tunable thresholds (tuned on train set, NOT test set)
    FUZZY_THRESHOLD = 0.72
    TFIDF_THRESHOLD = 0.45

    def __init__(self, ontology_path: str = None, log_experiments: bool = True):
        if ontology_path is None:
            ontology_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "ontology.json"
            )
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        self.log_experiments = log_experiments
        self._build_indices()
        self._build_tfidf()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_indices(self):
        """Build lookup tables for Layer 1 (exact + synonym match)."""
        self.exact_lookup: dict[str, dict] = {}
        self.all_terms: list[tuple[str, dict]] = []  # (normalized_term, skill_record)

        for skill in self.ontology:
            # Index display name
            key = skill["display_name"].strip().lower()
            self.exact_lookup[key] = skill
            self.all_terms.append((key, skill))

            # Index all synonyms
            for syn in skill.get("synonyms", []):
                syn_key = syn.strip().lower()
                self.exact_lookup[syn_key] = skill
                self.all_terms.append((syn_key, skill))

    def _build_tfidf(self):
        """Load trained TF-IDF vectorizer and matrix from disk."""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "trained_model.pkl"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Please run `python src/train.py` first."
            )
        import pickle
        with open(model_path, "rb") as f:
            model_artifacts = pickle.load(f)
            
        self.vectorizer = model_artifacts["vectorizer"]
        self.tfidf_matrix = model_artifacts["tfidf_matrix"]
        self.tfidf_skills = model_artifacts["tfidf_skills"]

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(raw_term: str) -> str:
        """Clean a raw term for matching."""
        if not raw_term or not isinstance(raw_term, str):
            return ""
        text = raw_term.strip()
        # Remove experience suffixes like "(3 yrs)", "- 5 years", "2+ years experience"
        text = re.sub(r'\(?\d+\+?\s*(?:yrs?|years?)?\s*(?:experience|exp)?\)?', '', text, flags=re.IGNORECASE)
        # Remove role prefixes/suffixes
        text = re.sub(r'\b(?:sr\.?|senior|junior|jr\.?|lead|staff|principal)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:developer|dev|engineer|architect|consultant|specialist)\b', '', text, flags=re.IGNORECASE)
        # Remove common separators
        text = re.sub(r'[/\\|,;]', ' ', text)
        # Remove "full-stack" etc
        text = re.sub(r'\b(?:full[\s-]?stack)\b', '', text, flags=re.IGNORECASE)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    @staticmethod
    def _is_noise(raw_term: str) -> bool:
        """Quick check if a term is obviously parsing noise."""
        if not raw_term or not isinstance(raw_term, str):
            return True
        stripped = raw_term.strip()
        if len(stripped) <= 1:
            return True
        if stripped in {"", " ", "\t", "\n"}:
            return True
        # All punctuation / special chars
        if all(c in "•●–—·:;.,!@#$%^&*()[]{}|\\/<>~`\"'+=-_\t\n " for c in stripped):
            return True
        # Common non-skill headers
        noise_patterns = [
            r'^skills?\s*[:&]', r'^technical\s+skills?', r'^key\s+competenc',
            r'^professional\s+experience', r'^education', r'^contact',
            r'^curriculum\s+vitae', r'^page\s+\d', r'^references\s+available',
            r'^certifications?\s*:', r'^awards?\s*:', r'^responsibilities',
            r'^duties\s*:', r'^summary\s*:',
        ]
        for pat in noise_patterns:
            if re.search(pat, stripped, re.IGNORECASE):
                return True
        # Looks like a date/phone/email
        if re.match(r'^\d{4}$', stripped):  # just a year
            return True
        if re.match(r'^[\d\-().+\s]{7,}$', stripped):  # phone number
            return True
        if re.match(r'^[^@]+@[^@]+\.[^@]+$', stripped):  # email
            return True
        # GPA, degree
        if re.match(r'^(?:gpa|b\.?tech|m\.?s\.?|b\.?s\.?|ph\.?d)', stripped, re.IGNORECASE):
            return True
        # University name pattern
        if re.match(r'^university\s+of', stripped, re.IGNORECASE):
            return True
        return False

    # ------------------------------------------------------------------
    # Layer 1: Exact / synonym lookup
    # ------------------------------------------------------------------

    def _layer1_exact(self, preprocessed: str, raw: str) -> dict | None:
        """Exact match against synonym table."""
        # Try preprocessed term
        if preprocessed in self.exact_lookup:
            skill = self.exact_lookup[preprocessed]
            return {
                "canonical_id": skill["canonical_id"],
                "display_name": skill["display_name"],
                "confidence": 1.0,
                "method": "exact_synonym_lookup",
                "match_detail": f"exact match on '{preprocessed}'",
            }
        # Try raw term lowercased (before preprocessing removes context)
        raw_lower = raw.strip().lower()
        if raw_lower in self.exact_lookup:
            skill = self.exact_lookup[raw_lower]
            return {
                "canonical_id": skill["canonical_id"],
                "display_name": skill["display_name"],
                "confidence": 1.0,
                "method": "exact_synonym_lookup",
                "match_detail": f"exact match on '{raw_lower}'",
            }
        return None

    # ------------------------------------------------------------------
    # Layer 2: Fuzzy string matching
    # ------------------------------------------------------------------

    def _layer2_fuzzy(self, preprocessed: str) -> dict | None:
        """Fuzzy match using SequenceMatcher + token overlap."""
        if not preprocessed or len(preprocessed) < 2:
            return None

        best_score = 0.0
        best_skill = None
        best_method_detail = ""

        preprocessed_tokens = set(preprocessed.split())

        for term, skill in self.all_terms:
            # Sequence ratio (edit-distance-like)
            seq_ratio = SequenceMatcher(None, preprocessed, term).ratio()

            # Token overlap (Jaccard-ish)
            term_tokens = set(term.split())
            if preprocessed_tokens and term_tokens:
                overlap = len(preprocessed_tokens & term_tokens)
                union = len(preprocessed_tokens | term_tokens)
                token_score = overlap / union if union > 0 else 0.0
            else:
                token_score = 0.0

            # Combined score (weighted average)
            combined = 0.6 * seq_ratio + 0.4 * token_score

            if combined > best_score:
                best_score = combined
                best_skill = skill
                best_method_detail = f"seq_ratio={seq_ratio:.3f}, token_overlap={token_score:.3f}"

        if best_score >= self.FUZZY_THRESHOLD and best_skill:
            return {
                "canonical_id": best_skill["canonical_id"],
                "display_name": best_skill["display_name"],
                "confidence": round(min(best_score, 0.95), 3),
                "method": "fuzzy_match",
                "match_detail": f"fuzzy ({best_method_detail}, combined={best_score:.3f})",
            }
        return None

    # ------------------------------------------------------------------
    # Layer 3: TF-IDF cosine similarity
    # ------------------------------------------------------------------

    def _layer3_tfidf(self, preprocessed: str) -> dict | None:
        """TF-IDF cosine similarity for semantic near-misses."""
        if not preprocessed or len(preprocessed) < 2:
            return None

        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.vectorizer.transform([preprocessed])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        best_idx = scores.argmax()
        best_score = float(scores[best_idx])

        if best_score >= self.TFIDF_THRESHOLD:
            skill = self.tfidf_skills[best_idx]
            return {
                "canonical_id": skill["canonical_id"],
                "display_name": skill["display_name"],
                "confidence": round(min(best_score * 0.9, 0.9), 3),
                "method": "tfidf_cosine",
                "match_detail": f"TF-IDF cosine={best_score:.3f}",
            }
        return None

    # ------------------------------------------------------------------
    # Main mapping function
    # ------------------------------------------------------------------

    def map_term(self, raw_term: str) -> dict:
        """
        Map a single raw parsed skill term to a canonical ontology node.

        Returns:
            {raw, canonical_id, display_name, confidence, method, match_detail}
        """
        # Edge cases
        if not raw_term or not isinstance(raw_term, str) or not raw_term.strip():
            return {
                "raw": raw_term if raw_term else "",
                "canonical_id": "unmapped",
                "display_name": None,
                "confidence": 0.0,
                "method": "invalid_input",
                "match_detail": "empty or invalid input",
            }

        # Noise detection
        if self._is_noise(raw_term):
            return {
                "raw": raw_term,
                "canonical_id": "unmapped",
                "display_name": None,
                "confidence": 0.0,
                "method": "noise_filter",
                "match_detail": "identified as parsing artifact / non-skill noise",
            }

        preprocessed = self._preprocess(raw_term)

        # Guard: if preprocessing ate everything
        if not preprocessed or len(preprocessed) < 2:
            return {
                "raw": raw_term,
                "canonical_id": "unmapped",
                "display_name": None,
                "confidence": 0.0,
                "method": "empty_after_preprocess",
                "match_detail": f"preprocessing reduced '{raw_term}' to '{preprocessed}' — no content left",
            }

        # Layer 1: Exact / synonym
        result = self._layer1_exact(preprocessed, raw_term)
        if result:
            return {"raw": raw_term, **result}

        # Layer 2: Fuzzy
        result = self._layer2_fuzzy(preprocessed)
        if result:
            return {"raw": raw_term, **result}

        # Layer 3: TF-IDF
        result = self._layer3_tfidf(preprocessed)
        if result:
            return {"raw": raw_term, **result}

        # No match
        return {
            "raw": raw_term,
            "canonical_id": "unmapped",
            "display_name": None,
            "confidence": 0.0,
            "method": "no_confident_match",
            "match_detail": f"no layer exceeded confidence threshold (fuzzy>{self.FUZZY_THRESHOLD}, tfidf>{self.TFIDF_THRESHOLD})",
        }

    def map_batch(self, raw_terms: list[str]) -> list[dict]:
        """Map a batch of raw terms. Deduplicates internally for efficiency."""
        results = []
        cache: dict[str, dict] = {}
        for term in raw_terms:
            key = term.strip().lower() if isinstance(term, str) else str(term)
            if key not in cache:
                cache[key] = self.map_term(term)
            results.append(cache[key])
        return results

    # ------------------------------------------------------------------
    # Experiment logging
    # ------------------------------------------------------------------

    def log_experiment(self, metrics: dict, log_path: str = None):
        """Append experiment run to CSV log."""
        if log_path is None:
            log_path = os.path.join(
                os.path.dirname(__file__), "..", "experiment_log.csv"
            )

        file_exists = os.path.isfile(log_path)
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "fuzzy_threshold": self.FUZZY_THRESHOLD,
            "tfidf_threshold": self.TFIDF_THRESHOLD,
            **metrics,
        }

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    # Quick sanity check
    mapper = SkillMapper()
    test_terms = [
        "ReactJS", "react.js", "Reactjs 2 yrs", "ML", "machine learning",
        "Sr. Python Dev", "pytohn", "•", "skills:", "xkq7z",
    ]
    print("=" * 70)
    print("MAPPER SANITY CHECK")
    print("=" * 70)
    for term in test_terms:
        result = mapper.map_term(term)
        mapped = f"{result['display_name']} ({result['confidence']:.2f})" if result['canonical_id'] != 'unmapped' else "UNMAPPED"
        print(f"  '{term}' -> {mapped}  [{result['method']}]")
    print("=" * 70)
