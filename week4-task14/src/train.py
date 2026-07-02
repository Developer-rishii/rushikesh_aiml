"""
Training script for the Skills Ontology Mapper.
This explicitly trains the ML layers (TF-IDF Vectorizer) and saves the trained 
artifacts to disk, separating the 'training' phase from the 'inference' phase.
"""

import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def train_models(ontology_path: str = None, model_output_path: str = None):
    if ontology_path is None:
        ontology_path = os.path.join(os.path.dirname(__file__), "..", "data", "ontology.json")
    if model_output_path is None:
        model_output_path = os.path.join(os.path.dirname(__file__), "..", "data", "trained_model.pkl")

    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    print(f"Loading ontology with {len(ontology)} skills for training...")

    # Build corpus for TF-IDF
    tfidf_docs = []
    tfidf_skills = []
    for skill in ontology:
        # Document is the display name plus all synonyms
        doc = skill["display_name"] + " " + " ".join(skill.get("synonyms", []))
        tfidf_docs.append(doc.lower())
        tfidf_skills.append(skill)

    print("Training TF-IDF Vectorizer (Character n-grams 2-4)...")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(tfidf_docs)

    print("Model trained successfully. Saving artifacts...")
    model_artifacts = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_skills": tfidf_skills
    }

    with open(model_output_path, "wb") as f:
        pickle.dump(model_artifacts, f)

    print(f"Trained ML model saved to {model_output_path}")

if __name__ == "__main__":
    train_models()
