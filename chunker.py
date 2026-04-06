"""
Document loader and recursive text chunker.
Supports: PDF, Excel (.xlsx/.xls), CSV, TXT/MD
"""

import os
import hashlib
from typing import Optional
from dataclasses import dataclass, field

import pdfplumber
import pandas as pd
import tiktoken

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, EMBEDDING_MODEL


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            raw = f"{self.metadata.get('source', '')}-{self.text[:100]}"
            self.chunk_id = hashlib.md5(raw.encode()).hexdigest()


# ── Tokenizer ───────────────────────────────────────────

_encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


# ── Loaders ─────────────────────────────────────────────

def load_pdf(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                if table:
                    header = table[0]
                    for row in table[1:]:
                        pairs = [
                            f"{h}: {v}" for h, v in zip(header, row)
                            if h and v
                        ]
                        table_text += " | ".join(pairs) + "\n"
            combined = text
            if table_text:
                combined += f"\n[TABLE DATA]\n{table_text}"
            pages.append(f"[Page {i + 1}]\n{combined}")
    return "\n\n".join(pages)


def load_excel(path: str) -> str:
    xls = pd.ExcelFile(path)
    parts = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df = df.dropna(how="all")
        if df.empty:
            continue
        rows = []
        cols = list(df.columns)
        for _, row in df.iterrows():
            pairs = [f"{c}: {row[c]}" for c in cols if pd.notna(row[c])]
            rows.append(" | ".join(pairs))
        parts.append(f"[Sheet: {sheet}]\n" + "\n".join(rows))
    return "\n\n".join(parts)


def load_csv(path: str) -> str:
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    rows = []
    cols = list(df.columns)
    for _, row in df.iterrows():
        pairs = [f"{c}: {row[c]}" for c in cols if pd.notna(row[c])]
        rows.append(" | ".join(pairs))
    return "\n".join(rows)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


LOADERS = {
    ".pdf": load_pdf,
    ".xlsx": load_excel,
    ".xls": load_excel,
    ".csv": load_csv,
    ".txt": load_text,
    ".md": load_text,
}


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")
    return loader(path)


# ── Recursive Chunker ──────────────────────────────────

def recursive_split(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    separators: list[str] = None,
) -> list[str]:
    """Split text recursively by separators, respecting token limits."""
    if separators is None:
        separators = SEPARATORS

    if count_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []

    # find the best separator that actually appears in the text
    sep = separators[0] if separators else " "
    remaining_seps = separators[1:] if len(separators) > 1 else [" "]
    for s in separators:
        if s in text:
            sep = s
            remaining_seps = separators[separators.index(s) + 1:] or [" "]
            break

    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current.strip():
                if count_tokens(current) <= max_tokens:
                    chunks.append(current.strip())
                else:
                    # recurse with finer separator
                    chunks.extend(recursive_split(current, max_tokens, overlap, remaining_seps))
            current = part

    if current.strip():
        if count_tokens(current) <= max_tokens:
            chunks.append(current.strip())
        else:
            chunks.extend(recursive_split(current, max_tokens, overlap, remaining_seps))

    # apply overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = _encoder.encode(chunks[i - 1])
            overlap_text = _encoder.decode(prev_tokens[-overlap:])
            overlapped.append(overlap_text + " " + chunks[i])
        chunks = overlapped

    return [c for c in chunks if c.strip()]


# ── Main Chunking Pipeline ─────────────────────────────

def chunk_document(
    path: str,
    tags: Optional[str] = None,
) -> list[Chunk]:
    """Load a file, chunk it, and attach metadata to each chunk."""
    filename = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    text = load_document(path)

    if not text.strip():
        print(f"  [warn] No text extracted from {filename}")
        return []

    raw_chunks = recursive_split(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        meta = {
            "source": filename,
            "file_type": ext,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "token_count": count_tokens(chunk_text),
        }
        if tags:
            meta["tags"] = tags
        chunks.append(Chunk(text=chunk_text, metadata=meta))

    print(f"  [{filename}] {len(chunks)} chunks ({count_tokens(text)} tokens total)")
    return chunks
