# 🩺 Prescription Chatbot (RAG + MCP + Ollama)

A local **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from **medical prescription PDFs**.  
Built with **FAISS** (for vector search), **Sentence-Transformers** (for embeddings), and **Ollama MCP models** (local LLMs).

⚠️ **Disclaimer**: This is for educational purposes only. Not a substitute for professional medical advice.

---

## 🚀 Features
- Preprocess PDFs once into FAISS index
- Local embeddings (no API costs)
- Local LLM inference with Ollama (`phi3:mini` recommended for low RAM)
- FastAPI backend with Swagger UI (`/docs`)
- Streamlit frontend for chat
- Source citations with expandable chunks

---

## 📂 Project Structure
rag_mcp_app/
│── app.py # FastAPI backend
│── retriever.py # FAISS retriever
│── ingestion.py # PDF processing
│── preprocess.py # Build FAISS index from PDFs
│── ollama_mcp.py # Ollama interface
│── ui.py # Streamlit chat UI
│── requirements.txt # Dependencies
│── data/
│ ├── pdfs/ # Store PDFs here
│ ├── index.faiss # Vector index (after preprocessing)
│ ├── meta.pkl # Metadata (after preprocessing)