import os, uuid
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append({"page": i+1, "text": text})
    return texts

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_index(pdf_path):
    docs, metas = [], []
    pages = extract_text_from_pdf(pdf_path)
    for p in pages:
        for chunk in chunk_text(p["text"]):
            docs.append(chunk)
            metas.append({"page": p["page"], "chunk": chunk})
    
    embeddings = embed_model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            old_meta = pickle.load(f)
        metas = old_meta + metas
        embeddings = np.vstack([index.reconstruct_n(0, index.ntotal), embeddings])
    else:
        index = faiss.IndexFlatL2(dim)
    
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    return {"chunks": len(docs), "pages": len(pages)}
