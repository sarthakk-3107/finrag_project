"""
RAG answer generator. Retrieves relevant chunks, builds context,
and generates cited answers using GPT-4o-mini.
"""

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL, SYSTEM_PROMPT, MAX_CONTEXT_CHUNKS
from retriever import semantic_search, SearchResult


client = OpenAI(api_key=OPENAI_API_KEY)


def build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    if not results:
        return "No relevant documents found."

    sections = []
    for i, r in enumerate(results):
        sections.append(
            f"--- Chunk {i + 1} ---\n"
            f"Source: {r.source} | Chunk: {r.chunk_index} | Score: {r.score}\n"
            f"{r.text}\n"
        )
    return "\n".join(sections)


def generate_answer(
    question: str,
    top_k: int = MAX_CONTEXT_CHUNKS,
    filter_source: str = None,
    filter_file_type: str = None,
    filter_tags: str = None,
) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks via semantic search
    2. Build context
    3. Generate answer with citations
    """
    # retrieve
    results = semantic_search(
        query=question,
        top_k=top_k,
        filter_source=filter_source,
        filter_file_type=filter_file_type,
        filter_tags=filter_tags,
    )

    context = build_context(results)

    # generate
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer the question using only the context above. "
                "Cite sources using [Source: filename, chunk N] format."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": r.source,
                "chunk_index": r.chunk_index,
                "score": r.score,
                "preview": r.text[:200] + "..." if len(r.text) > 200 else r.text,
            }
            for r in results
        ],
        "total_chunks_retrieved": len(results),
        "model": CHAT_MODEL,
    }
