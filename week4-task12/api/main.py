"""
FastAPI application for Parsing v0.
Endpoints:
  POST /parse — parse arbitrary resume/JD text
  GET /parse/eval/{doc_id} — explainability walkthrough for a sample document
  GET /parse/report — full metrics JSON
  GET /parse/edge-cases — edge case handling evidence
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.pipeline import ParsingPipeline
from src.evaluator import run_evaluation, HARD_CASE_IDS

app = FastAPI(title="PlaceMux Parsing v0", version="0.1.0",
              description="Resume/JD skill extraction API — Parsing v0")

# Initialize pipeline
pipeline = None

@app.on_event("startup")
def startup():
    global pipeline
    pipeline = ParsingPipeline()
    print("Parsing v0 pipeline loaded successfully.")

@app.get("/")
def read_root():
    """Root endpoint providing a welcome message and a link to the docs."""
    return {
        "message": "Welcome to the PlaceMux Parsing v0 API.",
        "docs": "Visit /docs for the interactive API documentation."
    }


class ParseRequest(BaseModel):
    text: str
    doc_type: Optional[str] = "resume"  # "resume" or "jd"


@app.post("/parse")
def parse_text(request: ParseRequest):
    """Parse arbitrary resume/JD text and return structured skills."""
    result = pipeline.parse(request.text)
    return {
        "doc_type": request.doc_type,
        "status": result["status"],
        "skills": result["skills"],
        "num_skills_extracted": len(result["skills"]),
    }


@app.get("/parse/eval/{doc_id}")
def eval_document(doc_id: str):
    """Explainability walkthrough: hits + misses for a specific sample document."""
    # Search for the doc in sample data
    doc = None
    for data_file in ['data/resumes.json', 'data/jds.json']:
        if not os.path.exists(data_file):
            continue
        with open(data_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        for d in docs:
            if d['doc_id'] == doc_id:
                doc = d
                break
        if doc:
            break
    
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in sample data.")
    
    result = pipeline.parse(doc['text'], doc['ground_truth'])
    
    return {
        "doc_id": doc_id,
        "text": doc['text'],
        "ground_truth": doc['ground_truth'],
        "status": result["status"],
        "extracted_skills": result["skills"],
        "misses": result["misses"],
        "num_hits": len(result["skills"]),
        "num_misses": len(result["misses"]),
    }


@app.get("/parse/report")
def get_report():
    """Full metrics JSON: baseline vs rules-only vs full pipeline."""
    report = run_evaluation(pipeline)
    return report


@app.get("/parse/edge-cases")
def get_edge_cases():
    """Returns how each edge case type was actually handled — real output, not descriptions."""
    edge_case_results = {}
    
    # Load sample data
    all_docs = {}
    for data_file in ['data/resumes.json', 'data/jds.json']:
        if not os.path.exists(data_file):
            continue
        with open(data_file, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        for d in docs:
            all_docs[d['doc_id']] = d
    
    case_descriptions = {
        'junk': 'Junk/irrelevant text — should return no skills',
        'alias_only': 'Alias-only resume — should match via ontology aliases',
        'negation': 'Negation case — negated skills should NOT be extracted',
        'empty': 'Empty/malformed input — should return no_skills_found',
        'substring_trap': 'Substring trap — "R" in "Director" should not be extracted',
    }
    
    for case_name, case_ids in HARD_CASE_IDS.items():
        case_results = []
        for doc_id in case_ids:
            doc = all_docs.get(doc_id)
            if not doc:
                continue
            
            result = pipeline.parse(doc['text'], doc['ground_truth'])
            
            case_results.append({
                'doc_id': doc_id,
                'input_text': doc['text'],
                'ground_truth': doc['ground_truth'],
                'status': result['status'],
                'extracted_skills': [s['canonical_name'] for s in result['skills']],
                'misses': result['misses'],
                'explanations': [s['explanation'] for s in result['skills']],
            })
        
        edge_case_results[case_name] = {
            'description': case_descriptions.get(case_name, ''),
            'results': case_results,
        }
    
    return edge_case_results
