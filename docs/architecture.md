# Architecture

## System Overview

PASsistant is a Retrieval-Augmented Generation (RAG) chatbot built on LangGraph. It processes academic documents (PDF) through OCR, indexes them into a vector database, and answers student questions by retrieving relevant context and generating grounded responses.

## Component Diagram

```mermaid
flowchart TB
    subgraph API["API Layer"]
        FastAPI[FastAPI REST + WebSocket]
        Telegram[Telegram Bot Adapter]
    end

    subgraph Graph["LangGraph Workflow"]
        Router[Router Node]
        Retrieval[Retrieval Node]
        Response[Response Node]
        OutputGuard[Output Guard]
        DocProcessor[Document Processor]
        StudentHandler[Student Record Handler]
    end

    subgraph Services["Service Layer"]
        Intent[IntentClassifier]
        ResponseGen[ResponseGeneration]
        DocProcessing[DocumentProcessing]
        IngestionHealth[IngestionHealth]
    end

    subgraph Infra["Infrastructure"]
        Qdrant[(Qdrant)]
        Redis[(Redis)]
        LLM[OpenAI / OpenRouter]
        Zhipu[Zhipu AI GLM-4 OCR]
    end

    API --> Graph
    Graph --> Services
    Services --> Infra
```

## Request Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant IG as Input Guard
    participant R as Router
    participant RN as Retrieval Node
    participant LLM as LLM Provider
    participant Q as Qdrant
    participant OG as Output Guard

    U->>API: POST /chat
    API->>IG: validate(message)
    IG-->>API: safe ✓
    API->>R: classify intent
    R-->>API: query_document
    API->>RN: run(state)
    RN->>LLM: rewrite query
    LLM-->>RN: keyword query
    par Parallel Search
        RN->>Q: search(query_1)
        RN->>Q: search(query_2)
        RN->>Q: search(query_3)
    end
    Q-->>RN: candidates
    RN->>LLM: rerank candidates
    LLM-->>RN: scored results
    RN-->>API: retrieved_chunks
    API->>LLM: generate response
    LLM-->>API: answer
    API->>OG: filter(answer)
    OG-->>API: safe answer
    API-->>U: ChatResponse
```

## Key Design Decisions

### Hierarchical Parent-Child Chunking

Documents are parsed into a tree structure mirroring their logical hierarchy (chapters → sections → subsections). Child chunks (small, indexed in vector DB) point to parent chunks (larger, stored on disk). At retrieval time, child hits are hydrated with parent context for richer LLM input.

### Contextual Embedding

Child chunk text is prepended with its breadcrumb path before embedding. This ensures that a table chunk under "III.4. Program Studi Teknik Informatika > Semester V" encodes the prodi and semester context in its vector, even if the raw table only contains course codes.

### Hybrid Retrieval with Reranking

Three retrieval strategies are supported:
- **similarity** — Dense cosine similarity only
- **rrf** — Reciprocal Rank Fusion of dense + BM25 sparse vectors
- **reranker** — First-stage RRF/similarity candidates re-scored by a cross-encoder

### Parallel Multi-Query

Up to 3 query variants (original, LLM-rewritten, expanded) are searched in parallel via `asyncio.gather`, reducing retrieval latency from 3x to 1x the single-query time.

## Directory Structure

```
src/
├── agent.py                 # LangGraph app entry point
├── api/                     # FastAPI REST + WebSocket layer
│   ├── routes/              # Endpoint handlers
│   ├── models.py            # Pydantic request/response schemas
│   └── services.py          # API orchestration
├── config/                  # Settings and logging
├── eval/                    # RAGAS evaluation framework
│   └── ragas/               # Evaluator, CLI, reporting
├── graphs/                  # LangGraph workflow definition
├── guardrails/              # Input/output safety filters
├── services/                # Business logic (intent, response, ingestion)
├── telegram_bot/            # Telegram integration
└── utils/
    ├── nodes/               # LangGraph node implementations
    ├── tools/               # OCR, chunking, student tools
    └── vector_store/        # Qdrant operations (indexing, search, BM25)
```
