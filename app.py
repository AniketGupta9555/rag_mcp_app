from fastapi import FastAPI
from retriever import retrieve
from ollama_mcp import call_ollama

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "RAG MCP app ready. Ask questions at /chat"}

@app.post("/chat")
async def chat(question: str):
    results = retrieve(question, k=3)
    context = "\n".join([f"{r['file']} (page {r['page']}): {r['chunk']}" for r in results])
    prompt = f"""You are a medical assistant. 
Use only the context from prescriptions below. 
If unsure, say 'Not enough info in prescriptions.'

Question: {question}
Context:\n{context}
"""
    answer = call_ollama(prompt)
    return {"answer": answer, "sources": results}
