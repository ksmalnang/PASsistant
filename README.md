# PASsistant — Academic Services & Student Records Chatbot

A **LangGraph-powered RAG chatbot** for Universitas Pasundan's Faculty of Engineering that answers student questions about **academic policies**, **curriculum**, and **student records**. Features hierarchical document chunking with contextual embeddings, hybrid retrieval (dense + BM25) with cross-encoder reranking, and GLM-OCR for PDF ingestion.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Document Management](#document-management)
- [Evaluation](#evaluation)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Documentation](#documentation)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Features

- **Hierarchical RAG** — Documents are parsed into a tree structure (chapters → sections → subsections), with parent-child chunk relationships for precise retrieval
- **Contextual Embeddings** — Child chunks are embedded with breadcrumb context (section path) for better semantic matching
- **Hybrid Retrieval** — Dense vector search + BM25 sparse vectors fused with RRF, or cross-encoder reranking
- **GLM-OCR** — PDF documents are parsed page-by-page with layout-aware OCR preserving table structure
- **Multi-query Expansion** — Queries are rewritten and expanded (including semester numeral normalization) for higher recall
- **Guardrails** — Input injection detection + output PII masking + system prompt leak prevention
- **RAGAS Evaluation** — Automated RAG quality assessment with Faithfulness, Answer Relevancy, Context Precision, and Context Recall
- **Multi-channel** — REST API, WebSocket streaming, and Telegram bot integration

---

## Architecture

```mermaid
flowchart TD
    User([User Query]) --> InputGuard[Input Guard]
    InputGuard --> Router[Router Node]
    
    Router -->|upload| OCR[OCR Ingest]
    Router -->|student| Record[Record Lookup]
    Router -->|query_document| Retrieval[Retrieval Node]
    
    OCR --> Response[Response Generation]
    Record --> Response
    Retrieval --> Response
    
    Response --> OutputGuard[Output Guard]
    OutputGuard --> Result([Response])
```

**Retrieval Pipeline Detail:**

```mermaid
flowchart LR
    Q[User Question] --> Rewrite[LLM Rewrite]
    Rewrite --> Queries["[original, rewritten, expanded]"]
    Queries --> Gather["asyncio.gather()"]
    
    Gather --> S1[Embed + Qdrant + Rerank]
    Gather --> S2[Embed + Qdrant + Rerank]
    Gather --> S3[Embed + Qdrant + Rerank]
    
    S1 --> Merge[Merge & Dedupe]
    S2 --> Merge
    S3 --> Merge
    
    Merge --> Hydrate[Hydrate Parents]
    Hydrate --> TopK[Top-K Results]
    TopK --> LLM[LLM Response]
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker (for Qdrant & Redis)
- API keys: OpenAI-compatible provider (e.g. OpenRouter), Zhipu AI (GLM-4 OCR)

### Setup

```bash
git clone <repo-url>
cd student-records-chatbot

# Install dependencies
uv sync --dev

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run Infrastructure

```bash
# Qdrant vector database
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Redis (optional, for caching)
docker run -p 6379:6379 redis:7-alpine
```

### Launch

```bash
# REST API server
uv run uvicorn src.api:app --reload --port 8000

# Telegram bot (polling mode for dev)
uv run python -m src.telegram_bot.polling
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@Kurikulum_IF_2021.pdf" \
  -F "files=@Buku_Panduan_Akademik.pdf"
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (Qdrant + Redis status) |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/by-filename/{filename}` | Delete a document from index |
| `POST` | `/upload` | Upload and ingest documents |
| `POST` | `/chat` | Send a chat message |
| `POST` | `/chat/upload` | Chat with file attachment |
| `POST` | `/chat/stream` | Streaming chat (SSE) |
| `POST` | `/chat/upload/stream` | Streaming chat with file |
| `POST` | `/telegram/webhook` | Telegram webhook receiver |
| `WS` | `/ws/{thread_id}` | WebSocket streaming |

---

## Document Management

```bash
# List all ingested documents
curl http://localhost:8000/documents

# Delete a specific document (re-ingest after pipeline changes)
curl -X DELETE "http://localhost:8000/documents/by-filename/Kurikulum%20IF%202021.pdf"

# Re-upload
curl -X POST http://localhost:8000/upload -F "files=@Kurikulum_IF_2021.pdf"
```

---

## Evaluation

Run RAGAS evaluation to measure retrieval and generation quality:

```bash
# Live evaluation against current index
uv run python -m src.eval.ragas \
    --dataset src/eval/datasets/ragas_dataset.jsonl \
    --mode live

# From pre-computed fixtures (no API calls for pipeline)
uv run python -m src.eval.ragas \
    --dataset src/eval/datasets/ragas_dataset.jsonl \
    --mode fixture
```

See [docs/evaluation.md](docs/evaluation.md) for full evaluation guide.

---

## Environment Variables

See [`.env.example`](.env.example) for all available variables. Key ones:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | LLM provider API key (OpenRouter, OpenAI, etc.) |
| `OPENAI_BASE_URL` | LLM provider base URL |
| `LLM_MODEL` | Primary LLM model for responses |
| `EMBEDDING_MODEL` | Embedding model for vector search |
| `ZHIPU_API_KEY` | Zhipu AI key for GLM-4 OCR |
| `QDRANT_URL` | Qdrant vector database URL |
| `RETRIEVAL_STRATEGY` | `similarity`, `rrf`, or `reranker` |
| `RETRIEVAL_TOP_K` | Number of chunks retrieved per query |
| `RERANKER_MODEL` | Cross-encoder model (when strategy=reranker) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |

---

## Development

```bash
# Run all tests
uv run pytest

# Run specific test suite
uv run pytest tests/test_prod_readiness.py -v

# Format & lint
uv run ruff format .
uv run ruff check .

# Type check
uv run basedpyright src/
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture and pipeline design |
| [docs/retrieval-pipeline.md](docs/retrieval-pipeline.md) | Retrieval strategy, chunking, and indexing details |
| [docs/evaluation.md](docs/evaluation.md) | RAGAS evaluation guide |
| [docs/deployment.md](docs/deployment.md) | Deployment and infrastructure guide |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM | OpenAI-compatible (DeepSeek, GPT-4o, Claude, etc.) |
| OCR | GLM-4 Vision (Zhipu AI) |
| Vector DB | Qdrant (dense + sparse vectors) |
| Embeddings | Configurable (Qwen, OpenAI, etc.) |
| Reranker | Jina Reranker v2 / FastEmbed cross-encoder |
| API | FastAPI + Uvicorn |
| Caching | Redis |
| Evaluation | RAGAS 0.4.3 |
| Observability | LangSmith |
| Package Manager | UV |

---

## License

MIT
