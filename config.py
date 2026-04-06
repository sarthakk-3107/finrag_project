import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"

# ── ChromaDB ────────────────────────────────────────────
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "financial_docs"

# ── Chunking ────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 64        # overlap tokens between chunks
SEPARATORS = ["\n\n", "\n", ". ", " "]

# ── Retrieval ───────────────────────────────────────────
DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.25   # minimum similarity score to include

# ── Generation ──────────────────────────────────────────
MAX_CONTEXT_CHUNKS = 8
SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions using ONLY the provided context chunks from financial documents. 

Rules:
1. If the context does not contain enough information, say so clearly.
2. Cite your sources using [Source: filename, chunk N] format.
3. When dealing with numbers, be precise. Do not round unless asked.
4. If multiple documents conflict, highlight the discrepancy.
5. Keep answers concise but thorough."""
