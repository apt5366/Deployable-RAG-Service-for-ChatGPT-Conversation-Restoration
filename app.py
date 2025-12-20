# app.py
from fastapi import FastAPI
from pydantic import BaseModel

from service import query_chat_history

app = FastAPI(title="ChatGPT History RAG")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    answer = query_chat_history(
        question=req.question,
        top_k=req.top_k
    )
    return {
        "question": req.question,
        "answer": answer
    }
