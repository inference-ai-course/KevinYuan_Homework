from fastapi import FastAPI
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load saved data
with open("rag_data.pkl", "rb") as f:
    data = pickle.load(f)
    all_chunks = data["all_chunks"]
    index = data["index"]

model = SentenceTransformer('all-MiniLM-L6-v2')
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG Search API for arXiv cs.CL papers", "endpoint": "/search?q=your_query"}

@app.get("/search")
async def search(q: str, k: int = 3):
    """Search for top-k most relevant passages"""
    query_vector = model.encode([q])
    distances, indices = index.search(np.array(query_vector), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "distance": float(distances[0][i]),
            "text": all_chunks[idx]
        })
    
    return {"query": q, "num_results": len(results), "results": results}
