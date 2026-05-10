# PASsistant — Academic Services & Student Records Chatbot

A **LangGraph-powered chatbot** that answers student questions about **academic services** and **personal student records**. Supports document upload with **GLM-4 Vision OCR**, knowledge-base retrieval via **Qdrant**, and is deployment-ready for **LangSmith/LangGraph Platform**.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Architecture

```
src/
├── agent.py              # Entry point & compiled LangGraph app
├── api/                  # FastAPI server (REST + WebSocket)
│   ├── routes/
│   │   ├── chat.py       # Chat endpoint
│   │   ├── documents.py  # Document upload endpoint
│   │   ├── health.py     # Health check
│   │   ├── router.py     # Route registration
│   │   └── websocket.py  # WebSocket streaming
│   ├── helpers.py        # API utilities
│   ├── models.py         # Request/response schemas
│   ├── services.py       # Orchestration logic
│   └── sessions.py       # Session management
├── telegram_bot/         # Telegram adapter, formatting, polling
├── config/
│   ├── logging.py        # RFC 5424 logging
│   └── settings.py       # Pydantic settings
├── graphs/
│   └── workflow.py       # LangGraph graph definition
├── services/             # Business logic layer
│   ├── contracts.py      # Dependency contracts
│   ├── document_processing.py
│   ├── intent.py
│   ├── response_generation.py
│   ├── session_registry.py
│   └── student_records.py
└── utils/
    ├── cache.py           # Redis-backed caching
    ├── state.py           # Agent state schema
    ├── nodes/             # LangGraph node adapters
    └── tools/             # OCR, Qdrant, storage, student tools
```

**Graph flow:**

1. **Router** — classifies user intent (upload, student record, or general question)
2. **Process document** — runs GLM-4 Vision OCR on uploaded files, chunks text hierarchically, embeds child chunks, and upserts into Qdrant
3. **Handle student record** — queries/updates structured student data
4. **Retrieve** — searches Qdrant (BM25 + dense vector) with optional reranking
5. **Generate response** — LLM with full context
6. **Check errors** — error handling and retry logic

---

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Docker (for local Qdrant & Redis)
- API keys for OpenAI and Zhipu AI (GLM-4)

### Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd student-records-chatbot

# Create virtual environment & install deps
uv venv .venv --python 3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync --dev

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, ZHIPU_API_KEY, etc.)
```

### Run Infrastructure

```bash
# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Start Redis (optional, for caching)
docker run -p 6379:6379 redis:7-alpine
```

### Launch the App

```bash
# Interactive CLI
python -m src.agent

# REST API server
uvicorn src.api:app --reload --port 8000

# LangGraph dev server
langgraph dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Send a message |
| `POST` | `/chat/upload` | Chat with file upload |
| `POST` | `/telegram/webhook` | Telegram webhook |
| `POST` | `/upload` | Upload documents only |
| `WS` | `/ws/{session_id}` | WebSocket streaming |

```bash
# Quick examples
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my GPA?"}'

curl -X POST http://localhost:8000/chat/upload \
  -F "message=Process this transcript" \
  -F "files=@transcript.pdf"

python -m src.telegram_bot.polling
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Primary LLM model | `deepseek/deepseek-v4-flash:exacto` |
| `LLM_REASONING_ENABLED` | Enable provider reasoning controls | `true` |
| `LLM_REASONING_EFFORT` | Reasoning effort level | `medium` |
| `LLM_REASONING_MAX_TOKENS` | Max reasoning tokens (overrides effort) | — |
| `LLM_REASONING_EXCLUDE` | Suppress reasoning in output | `true` |
| `EMBEDDING_MODEL` | Embedding model for vector search | `qwen/qwen3-embedding-8b:nitro` |
| `ZHIPU_API_KEY` | Zhipu AI API key (GLM-4 OCR) | Required |
| `ZHIPU_BASE_URL` | Zhipu AI base URL | `https://open.bigmodel.cn/api/paas/v4` |
| `GLM_OCR_MODEL` | OCR vision model | `glm-4v-flash` |
| `QDRANT_URL` | Qdrant instance URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | — |
| `QDRANT_COLLECTION_NAME` | Vector collection name | `student_documents` |
| `VECTOR_SIZE` | Embedding dimension | `4096` |
| `RETRIEVAL_STRATEGY` | Ranking strategy (`similarity`, `rrf`, `reranker`) | `similarity` |
| `RETRIEVAL_TOP_K` | Number of parent chunks retrieved, rendered in response context, and cited | `5` |
| `RERANKER_MODEL` | Reranker model (if reranking enabled) | — |
| `RERANKER_BASE_URL` | Remote reranker endpoint | — |
| `RERANKER_API_KEY` | Remote reranker API key | — |
| `RERANKER_CANDIDATE_MULTIPLIER` | Overfetch multiplier before reranking | `6` |
| `REDIS_URL` | Redis URL (caching) | — |
| `REDIS_KEY_PREFIX` | Redis key namespace | `student-records-chatbot` |
| `REDIS_CACHE_TTL_SECONDS` | Cache TTL | `300` |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `true` |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | LangSmith API key | Required |
| `LANGSMITH_PROJECT` | LangSmith project name | `student-records-chatbot` |
| `APP_ENV` | Application environment | `development` |
| `DEBUG` | Debug mode | `false` |
| `LOG_LEVEL` | Log level | `INFO` |
| `DATA_DIR` | Base data directory | `data` |
| `TELEGRAM_ENABLED` | Enable Telegram integration | `false` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | — |
| `TELEGRAM_WEBHOOK_URL` | Public webhook URL | — |
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | Webhook secret token | — |
| `TELEGRAM_MAX_FILE_BYTES` | Max Telegram upload size in bytes | `20000000` |

## Telegram

Use webhook mode in production and polling mode for local development.

```bash
# Local development
python -m src.telegram_bot.polling

# Register webhook explicitly
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"${TELEGRAM_WEBHOOK_URL}\",
    \"secret_token\": \"${TELEGRAM_WEBHOOK_SECRET_TOKEN}\",
    \"allowed_updates\": [\"message\"]
  }"

curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

Group chats use a shared session per Telegram chat id: `telegram:{chat_id}`.

---

## Development

```bash
# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Format & lint
ruff format .
ruff check .

# Type check
basedpyright src/
```

---

## Deployment

**LangGraph Platform (LangSmith):**

```bash
langgraph login
langgraph deploy
```

The `langgraph.json` config is already set up pointing to `src/agent.py:compiled_app`.

**Docker:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM | OpenAI-compatible (configurable) |
| OCR | GLM-4 Vision (Zhipu AI) |
| Vector DB | Qdrant |
| Embeddings | Configurable |
| API | FastAPI + Uvicorn |
| Config | Pydantic Settings |
| Package Manager | UV |
| Caching | Redis |
| Observability | LangSmith |

---

## License

MIT
