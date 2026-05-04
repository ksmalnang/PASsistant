# Evaluation Guide

This document covers the RAGAS-based RAG evaluation system for the PASsistant chatbot.

## Overview

The evaluation system measures both **retrieval quality** and **generation quality** using the [RAGAS](https://docs.ragas.io/) framework. It complements the existing retrieval-only evaluator with LLM-judged metrics that score the full question → retrieval → answer pipeline.

## Metrics

### Tier 1 — Core Metrics (always run)

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (hallucination detection) |
| **Answer Relevancy** | Is the answer focused and relevant to the question? |
| **Context Precision** | Signal-to-noise ratio — are relevant docs ranked highest? |
| **Context Recall** | Did retrieval find all information needed to answer? |

### Tier 2 — Extended Metrics (optional)

| Metric | What it measures |
|--------|-----------------|
| **Factual Correctness** | Factual agreement between response and ground truth |
| **Semantic Similarity** | Embedding-based similarity between response and reference |
| **Context Entity Recall** | Are named entities from ground truth in retrieved context? |
| **Noise Sensitivity** | Resilience to irrelevant/noisy retrieved chunks |

## Quick Start

### 1. Install evaluation dependencies

```bash
uv pip install -e ".[eval]"
```

### 2. Run evaluation (live mode)

```bash
python -m src.eval.ragas_eval \
    --dataset tests/fixtures/ragas_eval_dataset.jsonl \
    --mode live \
    --k-eval 5 \
    --output reports/ragas_eval_report.json
```

### 3. Run with extended metrics

```bash
python -m src.eval.ragas_eval \
    --dataset tests/fixtures/ragas_eval_dataset.jsonl \
    --mode live \
    --metrics-tier extended \
    --output reports/ragas_eval_extended.json
```

### 4. Run from pre-computed fixtures

```bash
python -m src.eval.ragas_eval \
    --dataset tests/fixtures/ragas_eval_dataset.jsonl \
    --mode fixture \
    --fixture tests/fixtures/ragas_eval_responses.json \
    --output reports/ragas_eval_fixture.json
```

## Evaluation Dataset

The dataset is stored in JSONL format at `tests/fixtures/ragas_eval_dataset.jsonl`.

### Schema

Each line is a JSON object with these fields:

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `id` | `str` | ✅ | Unique sample identifier |
| `user_input` | `str` | ✅ | The user question |
| `reference` | `str` | ✅ | Ground truth answer |
| `reference_contexts` | `list[str]` | ❌ | Ideal context passages |
| `metadata` | `dict` | ❌ | Category, difficulty, language |

### Example

```json
{
  "id": "ragas_q01",
  "user_input": "Apa syarat mengikuti ujian sidang skripsi?",
  "reference": "Mahasiswa harus telah menyelesaikan minimal 144 SKS...",
  "reference_contexts": ["Persyaratan sidang skripsi meliputi..."],
  "metadata": {
    "category": "academic_policy",
    "difficulty": "medium",
    "language": "id"
  }
}
```

### Adding New Samples

1. Add a new line to `tests/fixtures/ragas_eval_dataset.jsonl`
2. Ensure all required fields are present
3. Validate with:

```bash
python scripts/build_ragas_dataset.py \
    --validate tests/fixtures/ragas_eval_dataset.jsonl --stats
```

### Dataset Categories

- **academic_policy** — Admission, graduation, grading policies
- **curriculum** — Course structure, prerequisites, schedules
- **student_conduct** — Rules, regulations, disciplinary procedures
- **cross-document** — Questions requiring information from multiple sources

## Evaluation Modes

### Live Mode

Runs the full RAG pipeline (retrieval + generation) for each sample against the current Qdrant corpus. Requires:

- Running Qdrant instance with ingested documents
- Valid OpenRouter API key
- Configured embedding model

### Fixture Mode

Uses pre-computed responses from a JSON file, skipping the pipeline execution. Useful for:

- Reproducing evaluation results
- Testing the evaluation framework itself
- Comparing different model/prompt versions

Fixture file format:

```json
{
  "ragas_q01": {
    "response": "Generated answer...",
    "retrieved_contexts": ["Context chunk 1...", "Context chunk 2..."]
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAGAS_LLM_MODEL` | Falls back to `LLM_MODEL` | LLM model for RAGAS judge |
| `RAGAS_EMBEDDING_MODEL` | Falls back to `EMBEDDING_MODEL` | Embedding model for semantic metrics |

### Report Format

The evaluation produces a JSON report with:

- **`aggregate_scores`** — Mean scores across all samples
- **`per_sample`** — Individual scores for each question
- **`config`** — Evaluation configuration used

## Relationship to Retrieval Evaluation

The RAGAS evaluation **complements** the existing retrieval-only evaluator (`app/eval/retrieval_eval.py`):

| Aspect | Retrieval Eval | RAGAS Eval |
|--------|---------------|------------|
| Scope | Retrieval only | Full pipeline (retrieval + generation) |
| Metrics | Precision, recall, MRR | Faithfulness, relevancy, context quality |
| Judge | Exact match | LLM-as-judge |
| Cost | Free (no LLM calls) | LLM API costs for judging |
| Speed | Fast | Slower (LLM calls per sample × metric) |

Use both evaluations together for comprehensive pipeline assessment.
