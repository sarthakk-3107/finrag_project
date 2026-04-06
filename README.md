# FinRAG — Semantic Search & RAG over Financial Documents

A retrieval-augmented generation (RAG) system that ingests financial documents (PDFs, Excel, CSV, plain text), builds a semantic vector index using **ChromaDB** + **OpenAI embeddings**, and answers natural-language queries grounded in your documents.

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

## Features

- **Multi-format ingestion**: PDF, Excel (.xlsx/.xls), CSV, and plain text
- **Smart chunking**: Recursive text splitting with configurable overlap
- **Semantic search**: Cosine similarity retrieval via ChromaDB
- **Metadata filtering**: Filter by source file, document type, or custom tags
- **Cited answers**: LLM responses include source chunk references
- **FastAPI server**: REST API with Swagger docs
- **CLI mode**: Quick queries from the terminal

## Setup

```bash
# 1. Clone and navigate
cd finrag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Usage

### Ingest documents
```bash
# Ingest a single file
python ingest.py --file ./data/invoice_q3.pdf

# Ingest an entire folder
python ingest.py --dir ./data/

# Ingest with custom tags
python ingest.py --file ./data/revenue.xlsx --tags "quarterly,revenue,2024"
```

### Query via CLI
```bash
python query.py "What was the total revenue in Q3?"
python query.py "Which vendor had the highest invoice amount?"
python query.py "Summarize all outstanding payments"
```

### Query via API
```bash
# Start the server
python server.py

# Hit the endpoints (Swagger docs at http://localhost:8000/docs)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in Q3?"}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "outstanding invoices", "top_k": 5}'
```

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
├── .env                # Your OPENAI_API_KEY
└── data/               # Drop your financial docs here
    └── sample.csv      # Sample data included
```
