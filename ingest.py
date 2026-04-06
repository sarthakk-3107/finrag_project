"""
CLI tool to ingest financial documents into the vector store.

Usage:
    python ingest.py --file ./data/invoice.pdf
    python ingest.py --dir ./data/
    python ingest.py --file report.xlsx --tags "quarterly,revenue"
    python ingest.py --reset   # clear the vector store
"""

import os
import argparse
import time

from chunker import chunk_document, LOADERS
from embeddings import upsert_chunks, get_stats, reset_collection


def ingest_file(path: str, tags: str = None):
    print(f"\nIngesting: {path}")
    start = time.time()
    chunks = chunk_document(path, tags=tags)
    if chunks:
        upsert_chunks(chunks)
    elapsed = round(time.time() - start, 2)
    print(f"  Done in {elapsed}s")


def ingest_directory(dir_path: str, tags: str = None):
    supported = set(LOADERS.keys())
    files = []
    for root, _, filenames in os.walk(dir_path):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported:
                files.append(os.path.join(root, f))

    if not files:
        print(f"No supported files found in {dir_path}")
        print(f"Supported formats: {supported}")
        return

    print(f"\nFound {len(files)} files to ingest:")
    for f in files:
        print(f"  - {f}")

    for f in files:
        ingest_file(f, tags=tags)


def main():
    parser = argparse.ArgumentParser(description="Ingest financial documents into FinRAG")
    parser.add_argument("--file", type=str, help="Path to a single file")
    parser.add_argument("--dir", type=str, help="Path to a directory of files")
    parser.add_argument("--tags", type=str, help="Comma-separated tags (e.g., 'quarterly,revenue')")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    args = parser.parse_args()

    if args.reset:
        reset_collection()
        return

    if args.stats:
        stats = get_stats()
        print(f"\nCollection: {stats['collection']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Sample sources: {stats['sample_sources']}")
        return

    if args.file:
        if not os.path.isfile(args.file):
            print(f"File not found: {args.file}")
            return
        ingest_file(args.file, tags=args.tags)
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Directory not found: {args.dir}")
            return
        ingest_directory(args.dir, tags=args.tags)
    else:
        parser.print_help()
        return

    # show stats after ingest
    stats = get_stats()
    print(f"\nCollection now has {stats['total_chunks']} total chunks")


if __name__ == "__main__":
    main()
