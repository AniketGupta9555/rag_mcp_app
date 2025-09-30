import os, pickle, faiss, uuid
from ingestion import extract_text_from_pdf, chunk_text
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_DIR = "data/pdfs"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_pdfs():
    docs, metas = [], []

    # read all PDFs in data/pdfs
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".pdf"):
            path = os.path.join(DATA_DIR, fname)
            pages = extract_text_from_pdf(path)
            for p in pages:
                chunks = chunk_text(p["text"])
                for c in chunks:
                    docs.append(c)
                    metas.append({"file": fname, "page": p["page"], "chunk": c})

    # embed all chunks
    embeddings = embed_model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]

    # build faiss index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # save
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    print(f"✅ Preprocessed {len(docs)} chunks from {len(os.listdir(DATA_DIR))} PDFs.")

if __name__ == "__main__":
    preprocess_pdfs()
