"""
OpenAI embeddings + ChromaDB vector store operations.
Handles batch embedding, upsert, and collection management.
"""

from openai import OpenAI

import chromadb
from chromadb.config import Settings

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
)
from chunker import Chunk


# ── OpenAI Client ───────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts using OpenAI, batching to avoid limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def embed_single(text: str) -> list[float]:
    """Embed a single query string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ── ChromaDB Client ────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(chroma_client: chromadb.PersistentClient = None):
    if chroma_client is None:
        chroma_client = get_chroma_client()
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Upsert Chunks ──────────────────────────────────────

def upsert_chunks(chunks: list[Chunk], batch_size: int = 50):
    """Embed and upsert chunks into ChromaDB."""
    if not chunks:
        print("  No chunks to upsert.")
        return

    collection = get_collection()
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # embed in batches
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts, batch_size=batch_size)

    # upsert in batches
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"  Upserted {len(chunks)} chunks into '{COLLECTION_NAME}'")
    print(f"  Collection now has {collection.count()} total chunks")


# ── Collection Stats ───────────────────────────────────

def get_stats() -> dict:
    collection = get_collection()
    count = collection.count()
    sample = collection.peek(limit=3)
    sources = set()
    if sample and sample.get("metadatas"):
        for m in sample["metadatas"]:
            if m and m.get("source"):
                sources.add(m["source"])
    return {
        "total_chunks": count,
        "collection": COLLECTION_NAME,
        "sample_sources": list(sources),
    }


def reset_collection():
    """Delete and recreate the collection."""
    chroma_client = get_chroma_client()
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted collection '{COLLECTION_NAME}'")
    except Exception:
        pass
    get_collection(chroma_client)
    print(f"  Created fresh collection '{COLLECTION_NAME}'")
