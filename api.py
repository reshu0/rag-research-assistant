from fastapi import FastAPI
from pydantic import BaseModel

from final_rag_pipeline import run_rag


app = FastAPI(
    title="Enterprise RAG API",
    description="Hybrid RAG with RRF and cross-encoder reranking",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    answer = run_rag(request.question)

    return {
        "answer": answer
    }