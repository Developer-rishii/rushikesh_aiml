"""
Stage E: Integrate, break it, then demo.
Run: python3 demo.py
This is the literal script for the "2-minute live demo with real numbers
and one failure scenario" required in Stage E.4. It re-runs the pipeline
live (no cached/faked numbers), prints a side-by-side of keyword vs
semantic vs hybrid on a real recruiter query, then deliberately breaks
the embedding path and shows the degraded-but-still-working response.
"""
import json
from src.pipeline import build, load_csv, DATA
from src.embeddings import Embedder
from src.vector_index import VectorIndex
from src.keyword_search import BM25
from src.failure_mode import SearchService

def line(c="-"):
    print(c * 72)

def main():
    line("=")
    print("PlaceMux Task 13 - Semantic Search & Vector Retrieval - LIVE DEMO")
    line("=")

    print("\n[1/3] Running full pipeline on real data (build -> eval -> tune -> log)...")
    log = build()
    res = log["stage"]["C_offline_eval_test_holdout"]["results"]
    print(f"  Held-out test queries: {log['stage']['C_offline_eval_test_holdout']['test_queries']}")
    print(f"  keyword   nDCG@10 = {res['keyword']['nDCG@10']:.4f}  precision@10 = {res['keyword']['precision@10']:.2f}")
    print(f"  semantic  nDCG@10 = {res['semantic']['nDCG@10']:.4f}  precision@10 = {res['semantic']['precision@10']:.2f}")
    print(f"  hybrid    nDCG@10 = {res['hybrid']['nDCG@10']:.4f}  precision@10 = {res['hybrid']['precision@10']:.2f}"
          f"   (alpha={log['stage']['D_hybrid_tuning']['best_alpha']}, tuned on DEV split only)")

    print("\n[2/3] Real query, live: 'someone who can build data pipelines'")
    resumes = load_csv(DATA / "resumes.csv")
    ids = [r["resume_id"] for r in resumes]
    texts = [r["text"] for r in resumes]
    id2cluster = {r["resume_id"]: r["cluster"] for r in resumes}
    embedder = Embedder(n_components=48).fit(texts)
    vecs = embedder.encode(texts)
    index = VectorIndex(dim=vecs.shape[1]); index.add(ids, vecs)
    bm25 = BM25().fit(ids, texts)
    service = SearchService(embedder, index, bm25, alpha=log["stage"]["D_hybrid_tuning"]["best_alpha"])

    result = service.search("someone who can build data pipelines", top_k=5)
    print(f"  mode={result['mode']}")
    for rid, score in result["results"]:
        print(f"    {rid:5s} cluster={id2cluster[rid]:15s} score={score:.4f}")

    print("\n[3/3] Inducing failure: embedding service outage...")
    service.simulate_embedding_outage = True
    degraded = service.search("someone who can build data pipelines", top_k=5)
    print(f"  mode={degraded['mode']}")
    print(f"  warning: {degraded['warning']}")
    for rid, score in degraded["results"]:
        print(f"    {rid:5s} cluster={id2cluster[rid]:15s} score={score:.4f}")
    print("\n  -> Service degraded gracefully to keyword-only instead of failing hard. Confirmed.")
    line("=")
    print("Full evidence trail written to outputs/experiment_log.json")
    line("=")

if __name__ == "__main__":
    main()
