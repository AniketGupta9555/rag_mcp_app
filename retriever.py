import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=5):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = [metas[i] for i in I[0]]
    return results
