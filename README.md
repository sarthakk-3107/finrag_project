# finrag_project

# FinRAG — Semantic Search & RAG over Financial Documents

A retrieval-augmented generation (RAG) system that ingests financial documents (PDFs, Excel, CSV, plain text), builds a semantic vector index using **ChromaDB** + **OpenAI embeddings**, and answers natural-language queries grounded in your documents.

---

## Architecture

```
                  ┌──────────────┐
                  │  PDF / XLSX  │
                  │  CSV / TXT   │
                  └──────┬───────┘
                         │ ingest & chunk
                         ▼
              ┌─────────────────────┐
              │  Chunking Engine    │
              │  (recursive split)  │
              └──────────┬──────────┘
                         │ embed (text-embedding-3-small)
                         ▼
              ┌─────────────────────┐
              │     ChromaDB        │
              │  (vector store)     │
              └──────────┬──────────┘
                         │ semantic retrieval
                         ▼
              ┌─────────────────────┐
              │   GPT-4o-mini       │
              │  (answer + cite)    │
              └─────────────────────┘
```

---

## Features

- **Multi-format ingestion** — PDF (with table extraction), Excel (multi-sheet), CSV, and plain text
- **Smart chunking** — Recursive token-aware text splitting with configurable overlap
- **Semantic search** — Cosine similarity retrieval via ChromaDB with similarity thresholding
- **Metadata filtering** — Filter by source file, document type, or custom tags
- **Cited answers** — LLM responses include chunk-level source references
- **FastAPI server** — REST API with Swagger docs at `/docs`
- **CLI mode** — Quick queries and ingestion from the terminal

---

## Setup (Windows)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/finrag.git
cd finrag

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
set OPENAI_API_KEY=sk-your-key-here

# 5. Create the data folder and add your documents
mkdir data
# Place your PDF, Excel, CSV, or TXT files inside the data\ folder
```

## Setup (macOS / Linux)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/finrag.git
cd finrag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# 5. Create the data folder and add your documents
mkdir data
# Place your PDF, Excel, CSV, or TXT files inside the data/ folder
```

---

## Usage

### Ingest documents

```bash
# Ingest a single file
python ingest.py --file data\invoice_q3.pdf

# Ingest an entire folder
python ingest.py --dir data\

# Ingest with custom tags
python ingest.py --file data\revenue.xlsx --tags "quarterly,revenue,2024"

# Check collection stats
python ingest.py --stats

# Reset the vector store
python ingest.py --reset
```

### Query via CLI

```bash
# Full RAG (retrieval + LLM answer with citations)
python query.py "What was the total revenue in Q3?"
python query.py "Which vendor had the highest invoice amount?"
python query.py "Summarize all outstanding payments"

# Semantic search only (no LLM, just matching chunks)
python query.py "outstanding invoices" --search-only

# Filter by source or file type
python query.py "raw material costs" --source sample_invoices.csv
python query.py "quarterly trends" --type csv

# Output as JSON
python query.py "total operating expenses" --json

# List all ingested sources
python query.py --sources
```

### Query via API

```bash
# Start the server
python server.py

# Swagger docs at http://localhost:8000/docs

# Full RAG query
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What was the total revenue in Q3?\"}"

# Semantic search only
curl -X POST http://localhost:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"outstanding invoices\", \"top_k\": 5}"

# List sources
curl http://localhost:8000/sources

# Collection stats
curl http://localhost:8000/stats
```

---

## API Endpoints

| Method | Endpoint    | Description                              |
|--------|-------------|------------------------------------------|
| POST   | `/query`    | Full RAG: retrieve + generate answer     |
| POST   | `/search`   | Semantic search only, no LLM             |
| POST   | `/ingest`   | Ingest a document by file path           |
| GET    | `/sources`  | List all ingested source filenames       |
| GET    | `/stats`    | Collection statistics                    |
| POST   | `/reset`    | Reset the entire vector store            |

---

## Project Structure

```
finrag/
├── config.py           # Settings and constants
├── chunker.py          # Document loading + recursive chunking
├── embeddings.py       # OpenAI embedding wrapper + ChromaDB ops
├── retriever.py        # Semantic search + metadata filtering
├── generator.py        # RAG answer generation with citations
├── ingest.py           # CLI: ingest documents into vector store
├── query.py            # CLI: query the system
├── server.py           # FastAPI REST API
├── requirements.txt
├── .env.example        # API key template
├── .gitignore
└── data/               # Drop your financial docs here
    ├── sample_invoices.csv
    └── quarterly_revenue.csv
```

---

## How It Works

1. **Ingest** — Documents are loaded (PDF tables extracted, Excel sheets parsed, CSV rows formatted), then recursively split into token-aware chunks with overlap to preserve context.

2. **Embed** — Each chunk is embedded using OpenAI's `text-embedding-3-small` model and stored in ChromaDB with metadata (source file, file type, chunk index, custom tags).

3. **Retrieve** — User queries are embedded and matched against stored chunks using cosine similarity. Results below the similarity threshold are filtered out. Optional metadata filters narrow results by source, file type, or tags.

4. **Generate** — Top matching chunks are assembled into a context block and passed to GPT-4o-mini with a system prompt that enforces citation and precision. The response includes source references.

---

## Configuration

All settings are in `config.py`:

| Setting              | Default                  | Description                        |
|----------------------|--------------------------|------------------------------------|
| `EMBEDDING_MODEL`    | text-embedding-3-small   | OpenAI embedding model             |
| `CHAT_MODEL`         | gpt-4o-mini              | LLM for answer generation          |
| `CHUNK_SIZE`         | 512                      | Max tokens per chunk               |
| `CHUNK_OVERLAP`      | 64                       | Overlap tokens between chunks      |
| `DEFAULT_TOP_K`      | 5                        | Default number of results          |
| `SIMILARITY_THRESHOLD` | 0.25                   | Minimum cosine similarity to include |

---

## Tech Stack

- **Python 3.9+**
- **ChromaDB** — Vector store with cosine similarity search
- **OpenAI API** — Embeddings (text-embedding-3-small) and generation (GPT-4o-mini)
- **FastAPI** — REST API server
- **pdfplumber** — PDF text and table extraction
- **pandas** — Excel and CSV parsing
- **tiktoken** — Token counting for chunk sizing
