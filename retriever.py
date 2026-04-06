"""
Semantic retriever with metadata filtering and similarity thresholding.
"""

from dataclasses import dataclass

from embeddings import embed_single, get_collection
from config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    chunk_id: str

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", -1)

    def citation(self) -> str:
        return f"[Source: {self.source}, chunk {self.chunk_index}]"


def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filter_source: str = None,
    filter_file_type: str = None,
    filter_tags: str = None,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[SearchResult]:
    """
    Search the vector store using cosine similarity.
    Optionally filter by source filename, file type, or tags.
    """
    collection = get_collection()

    if collection.count() == 0:
        print("  [warn] Collection is empty. Ingest documents first.")
        return []

    # build metadata filter
    where_filter = _build_filter(filter_source, filter_file_type, filter_tags)

    # embed the query
    query_embedding = embed_single(query)

    # query ChromaDB
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    # parse results
    search_results = []
    if results and results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; similarity = 1 - distance
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            if similarity < threshold:
                continue

            search_results.append(
                SearchResult(
                    text=results["documents"][0][i],
                    score=round(similarity, 4),
                    metadata=results["metadatas"][0][i],
                    chunk_id=chunk_id,
                )
            )

    # sort by score descending
    search_results.sort(key=lambda r: r.score, reverse=True)
    return search_results


def _build_filter(source: str, file_type: str, tags: str) -> dict | None:
    conditions = []
    if source:
        conditions.append({"source": {"$eq": source}})
    if file_type:
        ft = file_type if file_type.startswith(".") else f".{file_type}"
        conditions.append({"file_type": {"$eq": ft}})
    if tags:
        conditions.append({"tags": {"$contains": tags}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def get_available_sources() -> list[str]:
    """List all unique source filenames in the collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    # peek at all (up to 10k)
    all_data = collection.get(limit=10000, include=["metadatas"])
    sources = set()
    for m in all_data["metadatas"]:
        if m and m.get("source"):
            sources.add(m["source"])
    return sorted(sources)
