"""
Pytest edge-case tests for Parsing v0.
Each test proves that a specific edge case is handled in real code.
"""
import os
import sys
import json
import pytest
import tempfile
import csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.ontology import SkillsOntology, OntologyError
from src.pipeline import ParsingPipeline
from src.rule_extractor import extract_candidates


# ---- Fixtures ----

@pytest.fixture(scope='module')
def pipeline():
    """Load the full pipeline once for all tests."""
    return ParsingPipeline(
        ontology_path='data/skills_ontology.csv',
        model_path='src/models/skill_classifier.pkl'
    )


@pytest.fixture(scope='module')
def ontology():
    """Load the ontology once for all tests."""
    return SkillsOntology('data/skills_ontology.csv')


# ---- Test 1: Ontology Validation ----

class TestOntologyValidation:
    """Test that the pipeline fails loudly on malformed ontology files."""
    
    def test_ontology_validation_missing_columns(self, tmp_path):
        """Ontology with missing required columns should raise OntologyError."""
        bad_file = tmp_path / "bad_ontology.csv"
        with open(bad_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['canonical_name', 'category'])  # missing 'aliases'
            writer.writerow(['Python', 'language'])
        
        with pytest.raises(OntologyError, match="missing required columns"):
            SkillsOntology(str(bad_file))
    
    def test_ontology_validation_duplicate_canonicals(self, tmp_path):
        """Ontology with duplicate canonical names should raise OntologyError."""
        bad_file = tmp_path / "dupe_ontology.csv"
        with open(bad_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['canonical_name', 'category', 'aliases'])
            writer.writerow(['Python', 'language', 'python|py'])
            writer.writerow(['Python', 'language', 'python3'])  # duplicate
        
        with pytest.raises(OntologyError, match="Duplicate canonical names"):
            SkillsOntology(str(bad_file))
    
    def test_ontology_validation_empty_aliases(self, tmp_path):
        """Ontology with empty alias list should raise OntologyError."""
        bad_file = tmp_path / "empty_alias_ontology.csv"
        with open(bad_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['canonical_name', 'category', 'aliases'])
            writer.writerow(['Python', 'language', ''])  # empty aliases
        
        with pytest.raises(OntologyError, match="Empty alias"):
            SkillsOntology(str(bad_file))
    
    def test_ontology_validation_file_not_found(self):
        """Missing ontology file should raise OntologyError."""
        with pytest.raises(OntologyError, match="not found"):
            SkillsOntology('/nonexistent/path/ontology.csv')
    
    def test_ontology_validation_malformed(self, tmp_path):
        """Combined test: any malformation in the ontology causes a loud failure."""
        # Missing columns
        bad1 = tmp_path / "bad1.csv"
        with open(bad1, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'cat'])
            writer.writerow(['Python', 'lang'])
        with pytest.raises(OntologyError):
            SkillsOntology(str(bad1))


# ---- Test 2: Empty/Malformed Input ----

class TestEmptyMalformedInput:
    """Confirm that empty or garbage text returns explicit no_skills_found."""
    
    def test_empty_string(self, pipeline):
        result = pipeline.parse("")
        assert result['status'] == 'no_skills_found'
        assert result['skills'] == []
    
    def test_whitespace_only(self, pipeline):
        result = pipeline.parse("   \n\t  ")
        assert result['status'] == 'no_skills_found'
        assert result['skills'] == []
    
    def test_garbage_text(self, pipeline):
        result = pipeline.parse("@#$%^&*()_+ 12345 !!!???")
        assert result['status'] == 'no_skills_found'
        assert result['skills'] == []
    
    def test_none_input(self, pipeline):
        result = pipeline.parse(None)
        assert result['status'] == 'no_skills_found'
        assert result['skills'] == []
    
    def test_empty_malformed_input(self, pipeline):
        """Named test for report: empty/malformed input → no_skills_found."""
        for text in ["", None, "   ", "!@#$%^&*()", "asdfjkl; qwerty zxcvbn"]:
            result = pipeline.parse(text)
            assert result['status'] == 'no_skills_found', f"Failed for text: {repr(text)}"
            assert len(result['skills']) == 0, f"Skills found in garbage text: {repr(text)}"


# ---- Test 3: Negation Handling ----

class TestNegationHandling:
    """Test that negated skills are NOT extracted."""
    
    def test_no_experience_with_docker(self, pipeline):
        result = pipeline.parse("I have no experience with Docker.")
        extracted_names = [s['canonical_name'] for s in result['skills']]
        assert 'Docker' not in extracted_names, "Docker should NOT be extracted from 'no experience with Docker'"
    
    def test_not_familiar_with_aws(self, pipeline):
        result = pipeline.parse("I am not familiar with AWS but I use Python daily.")
        extracted_names = [s['canonical_name'] for s in result['skills']]
        assert 'AWS' not in extracted_names, "AWS should NOT be extracted from 'not familiar with AWS'"
        assert 'Python' in extracted_names, "Python SHOULD be extracted"
    
    def test_negation_handling(self, pipeline):
        """Named test for report: negation is handled."""
        text = "Experienced Python developer. I have no experience with Docker or Kubernetes. I am not familiar with AWS, but I am learning."
        result = pipeline.parse(text, ground_truth=["Python"])
        extracted_names = [s['canonical_name'] for s in result['skills']]
        
        assert 'Python' in extracted_names, "Python should be extracted"
        assert 'Docker' not in extracted_names, "Docker should NOT be extracted (negated)"
        assert 'Kubernetes' not in extracted_names, "Kubernetes should NOT be extracted (negated)"
        assert 'AWS' not in extracted_names, "AWS should NOT be extracted (negated)"


# ---- Test 4: Substring / False-Positive Trap ----

class TestSubstringFalsePositiveTrap:
    """Test that short ambiguous tokens are not falsely extracted."""
    
    def test_r_in_director(self, pipeline):
        """'R' inside 'Director' should NOT be extracted."""
        result = pipeline.parse("I was a Director of Engineering.")
        extracted_names = [s['canonical_name'] for s in result['skills']]
        assert 'R' not in extracted_names, "'R' should not be extracted from 'Director'"
    
    def test_r_in_r_and_d(self, pipeline):
        """'R' in 'R&D' should NOT be extracted as R the language."""
        result = pipeline.parse("Working in R&D department on innovative projects.")
        extracted_names = [s['canonical_name'] for s in result['skills']]
        assert 'R' not in extracted_names, "'R' should not be extracted from 'R&D'"
    
    def test_substring_false_positive_trap(self, pipeline):
        """Named test for report: substring traps are handled."""
        text = "I was a Director of Engineering at R&D corp. Currently working on a project with Python. I am not a fan of the letter R, but I love programming."
        result = pipeline.parse(text, ground_truth=["Python"])
        extracted_names = [s['canonical_name'] for s in result['skills']]
        
        assert 'Python' in extracted_names, "Python should be extracted"
        assert 'R' not in extracted_names, "'R' should NOT be falsely extracted from 'Director' or 'R&D'"


# ---- Test 5: Alias-Only Resume ----

class TestAliasOnlyResume:
    """Confirm a resume using only aliases still achieves reasonable recall."""
    
    def test_alias_only_resume(self, pipeline):
        """Named test for report: alias-only resume has reasonable recall."""
        text = "Senior Engineer with 5 years in ML and DL. Proficient in tf and k8s. I also use r-script and js for quick prototypes. I love working in a dev-ops culture."
        ground_truth = ["Machine Learning", "Deep Learning", "TensorFlow", "Kubernetes", "R", "JavaScript", "DevOps"]
        
        result = pipeline.parse(text, ground_truth)
        extracted_names = [s['canonical_name'] for s in result['skills']]
        
        # Calculate recall
        hits = sum(1 for gt in ground_truth if gt in extracted_names)
        recall = hits / len(ground_truth)
        
        assert recall >= 0.5, f"Alias-only resume recall is {recall:.2f}, expected >= 0.50 (got {hits}/{len(ground_truth)} hits: {extracted_names})"
        
        # Specifically check that at least ML and k8s are resolved via aliases
        assert 'Machine Learning' in extracted_names, "ML alias should resolve to Machine Learning"
        assert 'Kubernetes' in extracted_names, "k8s alias should resolve to Kubernetes"
