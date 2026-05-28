"""
speech_analyzer.py
==================
Evaluates the communication quality of a speech transcript using NLP models.
"""

import re
import warnings
import string
from typing import Dict, Any, List
from pathlib import Path

# Attempt imports with graceful degradation
try:
    import spacy
    from transformers import pipeline
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    import textstat
    DEPENDENCIES_LOADED = True
except ImportError as e:
    warnings.warn(f"SpeechAnalyzer missing dependency: {e}. Running in graceful degradation mode.")
    DEPENDENCIES_LOADED = False

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False


# Constants
FILLER_WORDS = {"um", "uh", "like", "basically", "you know", "actually", "literally"}
TECHNICAL_TERMS = {
    "algorithm", "array", "binary", "cache", "class", "compiler", "cpu", "database",
    "debug", "deploy", "dictionary", "function", "hash", "heap", "integer", "interface",
    "iteration", "kubernetes", "latency", "library", "linux", "loop", "memory", "method",
    "network", "node", "object", "oop", "parameter", "pointer", "polymorphism", "query",
    "recursion", "runtime", "server", "sql", "stack", "string", "thread", "tree", "variable",
    "complexity", "graph", "api", "rest", "json", "xml", "git", "commit", "merge", "branch",
    "docker", "container", "aws", "cloud", "azure", "gcp", "agile", "scrum", "sprint",
    "architecture", "framework", "backend", "frontend", "fullstack", "machine learning",
    "deep learning", "neural network", "transformer", "nlp", "computer vision", "dataset",
    "training", "testing", "validation", "overfitting", "underfitting", "gradient",
    "optimization", "parameter", "hyperparameter", "python", "java", "c++", "javascript",
    "dictionary", "list", "tuple", "set"
}

class SpeechAnalyzer:
    """Analyzes a speech transcript to score communication quality."""
    
    def __init__(self) -> None:
        """Initialize the SpeechAnalyzer and load required NLP models."""
        self.sentiment_pipeline = None
        self.zero_shot_pipeline = None
        self.nlp = None
        self.language_tool = None
        
        if not DEPENDENCIES_LOADED:
            return
            
        try:
            # Download necessary NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Load Spacy model, download if missing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                print("Downloading spacy model 'en_core_web_sm'...")
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                
            # Load Transformers
            print("Loading NLP pipelines (this may take a minute)...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            
        except Exception as e:
            warnings.warn(f"Failed to load some NLP models: {e}. Running with limited capabilities.")

        if LANGUAGE_TOOL_AVAILABLE:
            try:
                self.language_tool = language_tool_python.LanguageTool('en-US')
            except Exception as e:
                warnings.warn(f"Failed to load LanguageTool (Java may be missing): {e}")
                self.language_tool = None


    def analyze(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze the transcript and return all communication sub-scores and signals.
        
        Args:
            transcript (str): The text transcript of the candidate's speech.
            
        Returns:
            Dict[str, Any]: A dictionary containing scores and raw signals.
        """
        if not transcript or not transcript.strip():
            return {
                "fluency_score": 0.0,
                "clarity_score": 0.0,
                "vocabulary_score": 0.0,
                "confidence_score": 0.0,
                "grammar_score": 0.0,
                "communication_score": 0.0,
                "signals": {
                    "avg_sentence_length": 0.0,
                    "flesch_reading_ease": 0.0,
                    "flesch_kincaid_grade": 0.0,
                    "unique_word_ratio": 0.0,
                    "filler_word_count": 0,
                    "grammar_error_count": 0,
                    "sentiment_label": "UNKNOWN",
                    "sentiment_confidence": 0.0,
                    "tone_labels": {},
                    "technical_terms_count": 0,
                    "sentence_count": 0,
                    "word_count": 0
                }
            }

        signals: Dict[str, Any] = {}
        
        # Populate basic counts for signals
        sentences = sent_tokenize(transcript) if DEPENDENCIES_LOADED else [s for s in transcript.split(".") if s]
        words = word_tokenize(transcript) if DEPENDENCIES_LOADED else transcript.split()
        words = [w.lower() for w in words if w.isalnum()]
        
        signals["sentence_count"] = len(sentences)
        signals["word_count"] = len(words)
        signals["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0.0
        
        # Fluency
        fluency_score, filler_count = self._score_fluency(transcript, signals["avg_sentence_length"])
        signals["filler_word_count"] = filler_count
        
        # Clarity
        clarity_score, ease, grade = self._score_clarity(transcript)
        signals["flesch_reading_ease"] = ease
        signals["flesch_kincaid_grade"] = grade
        
        # Vocabulary
        vocab_score, unique_ratio, tech_count = self._score_vocabulary(words)
        signals["unique_word_ratio"] = unique_ratio
        signals["technical_terms_count"] = tech_count
        
        # Confidence
        confidence_score, sent_label, sent_conf, tone_dict = self._score_confidence(transcript)
        signals["sentiment_label"] = sent_label
        signals["sentiment_confidence"] = sent_conf
        signals["tone_labels"] = tone_dict
        
        # Grammar
        grammar_score, error_count = self._score_grammar(transcript)
        signals["grammar_error_count"] = error_count
        
        scores = {
            "fluency": fluency_score,
            "clarity": clarity_score,
            "vocabulary": vocab_score,
            "confidence": confidence_score,
            "grammar": grammar_score
        }
        
        comm_score = self._compute_communication_score(scores)
        
        return {
            "fluency_score": round(fluency_score, 2),
            "clarity_score": round(clarity_score, 2),
            "vocabulary_score": round(vocab_score, 2),
            "confidence_score": round(confidence_score, 2),
            "grammar_score": round(grammar_score, 2),
            "communication_score": round(comm_score, 2),
            "signals": signals
        }

    def _score_fluency(self, transcript: str, avg_sentence_length: float) -> tuple[float, int]:
        """Score fluency based on filler words and sentence length."""
        lower_transcript = transcript.lower()
        filler_count = 0
        for filler in FILLER_WORDS:
            # Word boundary regex to catch exact words/phrases
            filler_count += len(re.findall(rf"\b{filler}\b", lower_transcript))
            
        score = 100.0
        # Penalize filler words (-2 pts each)
        score -= (filler_count * 2)
        
        # Penalize short sentences
        if 0 < avg_sentence_length < 5:
            score -= 10
        # Reward appropriate sentence length
        elif 15 <= avg_sentence_length <= 20:
            score += 10
            
        return max(0.0, min(100.0, score)), filler_count

    def _score_clarity(self, transcript: str) -> tuple[float, float, float]:
        """Score clarity using Flesch-Kincaid metrics and structural phrases."""
        if not DEPENDENCIES_LOADED:
            return 50.0, 0.0, 0.0
            
        try:
            ease = textstat.flesch_reading_ease(transcript)
            grade = textstat.flesch_kincaid_grade(transcript)
        except Exception:
            ease, grade = 0.0, 0.0

        # Normalize ease (usually 0-100, can be negative)
        score = max(0.0, min(100.0, ease))
        
        if grade > 16 or grade < 6:
            score -= 15
            
        # Reward structured responses
        structural_phrases = ["first", "then", "finally", "in summary", "to begin", "next", "therefore"]
        lower_transcript = transcript.lower()
        for phrase in structural_phrases:
            if re.search(rf"\b{phrase}\b", lower_transcript):
                score += 5
                
        return max(0.0, min(100.0, score)), ease, grade

    def _score_vocabulary(self, words: List[str]) -> tuple[float, float, int]:
        """Score vocabulary based on uniqueness and technical terms."""
        total_words = len(words)
        if total_words == 0:
            return 0.0, 0.0, 0
            
        unique_words = set(words)
        unique_ratio = len(unique_words) / total_words
        
        tech_count = sum(1 for w in words if w in TECHNICAL_TERMS)
        tech_density = tech_count / total_words
        
        # Normalize technical density (assuming 5% is a very high density in normal speech)
        tech_density_normalized = min(1.0, tech_density * 20) 
        
        score = (unique_ratio * 50) + (tech_density_normalized * 50)
        return max(0.0, min(100.0, score)), unique_ratio, tech_count

    def _score_confidence(self, transcript: str) -> tuple[float, str, float, Dict[str, float]]:
        """Score confidence using zero-shot classification and sentiment analysis."""
        score = 50.0
        sent_label = "UNKNOWN"
        sent_conf = 0.0
        tone_dict: Dict[str, float] = {}
        
        if self.sentiment_pipeline and self.zero_shot_pipeline:
            try:
                # Sentiment
                sent_result = self.sentiment_pipeline(transcript[:512])[0]
                sent_label = sent_result['label']
                sent_conf = sent_result['score']
                
                # Zero-shot Tone
                labels = ["confident", "hesitant", "uncertain", "assertive"]
                zs_result = self.zero_shot_pipeline(transcript[:512], candidate_labels=labels)
                
                for label, prob in zip(zs_result['labels'], zs_result['scores']):
                    tone_dict[label] = prob
                    
                # Calculate score
                conf_prob = tone_dict.get("confident", 0.0) + tone_dict.get("assertive", 0.0)
                hes_prob = tone_dict.get("hesitant", 0.0) + tone_dict.get("uncertain", 0.0)
                
                # Base score mapped from probability
                score = (conf_prob * 100) - (hes_prob * 50)
                
                # Boost if sentiment is POSITIVE
                if sent_label == "POSITIVE":
                    score += 10
            except Exception as e:
                warnings.warn(f"Confidence scoring failed: {e}")
                
        return max(0.0, min(100.0, score)), sent_label, sent_conf, tone_dict

    def _score_grammar(self, transcript: str) -> tuple[float, int]:
        """Score grammar using language_tool_python."""
        error_count = 0
        if self.language_tool:
            try:
                matches = self.language_tool.check(transcript)
                error_count = len(matches)
            except Exception as e:
                warnings.warn(f"Language check failed: {e}")
                
        score = max(0.0, 100.0 - (error_count * 5.0))
        return score, error_count

    def _compute_communication_score(self, scores: Dict[str, float]) -> float:
        """Compute the weighted average communication score."""
        weights = {
            "fluency": 0.25,
            "clarity": 0.25,
            "vocabulary": 0.20,
            "confidence": 0.20,
            "grammar": 0.10
        }
        total_score = sum(scores[k] * weights[k] for k in weights)
        return max(0.0, min(100.0, total_score))
