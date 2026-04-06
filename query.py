"""
CLI tool to query the FinRAG system.

Usage:
    python query.py "What was the total revenue in Q3?"
    python query.py "Which vendor has the highest outstanding balance?" --source invoices.csv
    python query.py "Summarize all payments" --type pdf --top_k 10
"""

import argparse
import json

from generator import generate_answer
from retriever import semantic_search, get_available_sources


def main():
    parser = argparse.ArgumentParser(description="Query the FinRAG system")
    parser.add_argument("question", type=str, nargs="?", help="Your question")
    parser.add_argument("--source", type=str, help="Filter by source filename")
    parser.add_argument("--type", type=str, help="Filter by file type (pdf, xlsx, csv)")
    parser.add_argument("--tags", type=str, help="Filter by tags")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--search-only", action="store_true", help="Only search, no LLM answer")
    parser.add_argument("--sources", action="store_true", help="List available sources")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    args = parser.parse_args()

    if args.sources:
        sources = get_available_sources()
        print("\nIngested sources:")
        for s in sources:
            print(f"  - {s}")
        return

    if not args.question:
        parser.print_help()
        return

    if args.search_only:
        # just semantic search, no generation
        results = semantic_search(
            query=args.question,
            top_k=args.top_k,
            filter_source=args.source,
            filter_file_type=args.type,
            filter_tags=args.tags,
        )
        print(f"\nFound {len(results)} results for: \"{args.question}\"\n")
        for i, r in enumerate(results):
            print(f"  [{i + 1}] Score: {r.score} | {r.citation()}")
            print(f"      {r.text[:200]}...")
            print()
        return

    # full RAG
    result = generate_answer(
        question=args.question,
        top_k=args.top_k,
        filter_source=args.source,
        filter_file_type=args.type,
        filter_tags=args.tags,
    )

    if args.as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n{'=' * 60}")
    print(f"Q: {result['question']}")
    print(f"{'=' * 60}")
    print(f"\n{result['answer']}\n")
    print(f"{'=' * 60}")
    print(f"Sources ({result['total_chunks_retrieved']} chunks retrieved):")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, score: {s['score']})")
    print()


if __name__ == "__main__":
    main()
