"""
FastAPI REST API for FinRAG.

Endpoints:
    POST /query     - Full RAG (retrieve + generate answer)
    POST /search    - Semantic search only (no LLM)
    POST /ingest    - Ingest a file by path
    GET  /sources   - List ingested sources
    GET  /stats     - Collection stats
    POST /reset     - Reset the vector store

Run:
    python server.py
    # Swagger docs at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os

from generator import generate_answer
from retriever import semantic_search, get_available_sources
from chunker import chunk_document
from embeddings import upsert_chunks, get_stats, reset_collection


app = FastAPI(
    title="FinRAG API",
    description="Semantic search and RAG over financial documents",
    version="1.0.0",
)


# ── Request / Response Models ──────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    filter_source: str | None = None
    filter_file_type: str | None = None
    filter_tags: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filter_source: str | None = None
    filter_file_type: str | None = None
    filter_tags: str | None = None


class IngestRequest(BaseModel):
    file_path: str
    tags: str | None = None


# ── Endpoints ──────────────────────────────────────────

@app.post("/query")
def query_endpoint(req: QueryRequest):
    """Full RAG: retrieve relevant chunks and generate a cited answer."""
    try:
        result = generate_answer(
            question=req.question,
            top_k=req.top_k,
            filter_source=req.filter_source,
            filter_file_type=req.filter_file_type,
            filter_tags=req.filter_tags,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """Semantic search only, no LLM generation."""
    try:
        results = semantic_search(
            query=req.query,
            top_k=req.top_k,
            filter_source=req.filter_source,
            filter_file_type=req.filter_file_type,
            filter_tags=req.filter_tags,
        )
        return {
            "query": req.query,
            "results": [
                {
                    "text": r.text,
                    "score": r.score,
                    "source": r.source,
                    "chunk_index": r.chunk_index,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    """Ingest a document by file path."""
    if not os.path.isfile(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    try:
        chunks = chunk_document(req.file_path, tags=req.tags)
        if chunks:
            upsert_chunks(chunks)
        return {
            "file": req.file_path,
            "chunks_created": len(chunks),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
def sources_endpoint():
    """List all ingested source filenames."""
    return {"sources": get_available_sources()}


@app.get("/stats")
def stats_endpoint():
    """Get collection statistics."""
    return get_stats()


@app.post("/reset")
def reset_endpoint():
    """Reset the entire vector store."""
    reset_collection()
    return {"status": "reset complete"}


# ── Run ────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
