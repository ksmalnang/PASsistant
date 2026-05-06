"""RAGAS-based RAG evaluation for the PASsistant pipeline (RAGAS v0.4.3+).

Supports two evaluation modes:
- **live**: runs the full RAG pipeline (retrieval + generation) for each sample
- **fixture**: loads pre-computed responses from a JSON file

Usage:
    python -m src.eval.ragas_eval \\
        --dataset tests/fixtures/ragas_dataset.jsonl \\
        --mode live \\
        --output reports/ragas_eval_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RagasEvalConfig:
    """Configuration for a RAGAS evaluation run."""

    dataset_path: Path
    output_path: Path | None = None
    metrics_tier: str = "core"  # "core" or "extended"
    mode: str = "live"  # "live" or "fixture"
    fixture_path: Path | None = None  # pre-computed responses (fixture mode)
    k_eval: int = 5
    batch_size: int = 1
    evaluation_date: str = field(default_factory=lambda: datetime.now(UTC).date().isoformat())
    openrouter_min_interval_seconds: float = 1.0
    openrouter_jitter_seconds: float = 0.25
    openrouter_max_retries: int = 3
    openrouter_backoff_base_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.output_path is None:
            reports_dir = Path("src") / "eval" / "reports"
            self.output_path = reports_dir / f"{self.evaluation_date}_ragas_eval_report.json"


@dataclass
class RagasEvalSample:
    """Single evaluation sample with all pipeline outputs."""

    id: str
    question: str
    ground_truth: str
    provided_contexts: list[str] = field(default_factory=list)
    reference_contexts: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Simple textual progress bar with ETA for sequential steps."""

    def __init__(self, total: int, label: str):
        self.total = max(total, 1)
        self.label = label
        self.completed = 0
        self.started_at = time.perf_counter()

    def advance(self, sample_id: str) -> None:
        self.completed += 1
        elapsed = max(time.perf_counter() - self.started_at, 0.001)
        avg_seconds = elapsed / self.completed
        remaining = max(self.total - self.completed, 0)
        eta_seconds = int(avg_seconds * remaining)
        bar_width = 20
        filled = int((self.completed / self.total) * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        logger.info(
            "%s [%s] %d/%d current=%s elapsed=%s eta=%s",
            self.label,
            bar,
            self.completed,
            self.total,
            sample_id,
            self._fmt(int(elapsed)),
            self._fmt(eta_seconds),
        )

    @staticmethod
    def _fmt(seconds: int) -> str:
        minutes, secs = divmod(max(seconds, 0), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[RagasEvalSample]:
    """Load and validate a JSONL evaluation dataset.

    Expected row schema:
    {
        "id": "...",
        "question": "...",
        "contexts": [{"text": "...", "is_relevant": true}],
        "answer": "...",
        "ground_truth": "...",
        "metadata": {...}
    }

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file is empty or any row is missing required fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[RagasEvalSample] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            required_fields = {"id", "question", "contexts", "answer", "ground_truth"}
            missing = required_fields - set(row.keys())
            if missing:
                raise ValueError(
                    f"Line {line_no} missing required field(s): {', '.join(sorted(missing))}"
                )

            sample_id = row["id"]
            question = row["question"]
            ground_truth = row["ground_truth"]
            answer = row["answer"]
            contexts = row["contexts"]
            if not isinstance(contexts, list):
                raise ValueError(f"Line {line_no}: contexts must be a list")

            provided_contexts: list[str] = []
            reference_contexts: list[str] = []
            for idx, context in enumerate(contexts, start=1):
                if not isinstance(context, dict):
                    raise ValueError(f"Line {line_no}: contexts[{idx}] must be an object")
                text = context.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Line {line_no}: contexts[{idx}].text must be a non-empty string"
                    )
                is_relevant = context.get("is_relevant")
                if is_relevant is not None and not isinstance(is_relevant, bool):
                    raise ValueError(
                        f"Line {line_no}: contexts[{idx}].is_relevant must be a boolean"
                    )
                provided_contexts.append(text)
                if context.get("is_relevant", True):
                    reference_contexts.append(text)

            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"Line {line_no}: id must be a non-empty string")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_no}: question must be a non-empty string")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Line {line_no}: answer must be a non-empty string")
            if not isinstance(ground_truth, str) or not ground_truth.strip():
                raise ValueError(f"Line {line_no}: ground_truth must be a non-empty string")
            if not provided_contexts:
                raise ValueError(f"Line {line_no}: contexts must contain at least one item")
            if not isinstance(reference_contexts, list):
                raise ValueError(f"Line {line_no}: reference_contexts must be a list")
            if not isinstance(provided_contexts, list):
                raise ValueError(f"Line {line_no}: provided contexts must be a list")
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Line {line_no}: metadata must be an object")

            samples.append(
                RagasEvalSample(
                    id=sample_id,
                    question=question,
                    ground_truth=ground_truth,
                    provided_contexts=provided_contexts,
                    reference_contexts=reference_contexts,
                    response=answer,
                    metadata=metadata,
                )
            )

    if not samples:
        raise ValueError(f"Dataset is empty: {path}")

    logger.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


def load_fixtures(path: Path) -> dict[str, dict[str, Any]]:
    """Load pre-computed pipeline responses for fixture mode.

    Expects a JSON file mapping sample IDs to their pipeline outputs::

        {
          "ragas_q01": {
            "response": "...",
            "retrieved_contexts": ["..."]
          }
        }

    Returns:
        Mapping of sample ID → pipeline output dict.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Fixture file must contain a JSON object mapping sample IDs to outputs")

    logger.info("Loaded fixtures for %d samples from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# RagasEvaluator
# ---------------------------------------------------------------------------


class RagasEvaluator:
    """Orchestrates RAGAS evaluation of the RAG pipeline."""

    def __init__(self, config: RagasEvalConfig):
        self.config = config
        self._llm: Any = None
        self._embeddings: Any = None
        self._retriever: Any = None
        self._retrieval_node: Any = None
        self._response_service: Any = None
        self._request_lock = threading.Lock()
        self._last_openrouter_request_at = 0.0
        self._usage_summary: dict[str, Any] = {
            "request_count": 0,
            "llm_request_count": 0,
            "embedding_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd_reported": 0.0,
            "cost_fields": [],
        }

    # -- public entry point --------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Execute the full evaluation pipeline and return a JSON report."""
        logger.info("Loading evaluation dataset from %s", self.config.dataset_path)
        samples = load_dataset(self.config.dataset_path)
        samples = await self._execute_pipeline(samples)
        results = self._evaluate_with_ragas(samples)
        report = self._build_report(samples, results)
        self._save_report(report)
        return report

    # -- pipeline execution --------------------------------------------------

    async def _execute_pipeline(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        """Populate `response` and `retrieved_contexts` for every sample."""
        if self.config.mode == "fixture":
            return self._apply_fixtures(samples)
        return await self._run_live_pipeline(samples)

    def _apply_fixtures(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        """Fill samples from a pre-computed fixture file or dataset-provided answers."""
        progress = ProgressTracker(len(samples), "Fixture")
        if self.config.fixture_path is None:
            for sample in samples:
                logger.info("Using dataset-provided fixture data for %s", sample.id)
                if sample.response:
                    sample.retrieved_contexts = list(sample.provided_contexts)
                progress.advance(sample.id)
            return samples

        fixtures = load_fixtures(self.config.fixture_path)
        for sample in samples:
            logger.info("Loading fixture output for %s", sample.id)
            fixture = fixtures.get(sample.id, {})
            sample.response = fixture.get("response", "")
            sample.retrieved_contexts = fixture.get("retrieved_contexts", [])
            progress.advance(sample.id)
        return samples

    async def _run_live_pipeline(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        """Run the real RAG pipeline for each sample."""
        self._init_pipeline_services()
        progress = ProgressTracker(len(samples), "Pipeline")

        for sample in samples:
            try:
                logger.info("Running live pipeline for %s", sample.id)
                from langchain_core.messages import HumanMessage

                from src.utils.state import AgentState

                state = AgentState(
                    messages=[HumanMessage(content=sample.question)],
                    requires_retrieval=True,
                    retrieval_query=sample.question,
                    current_intent="query_document",
                )
                retrieval_updates = await self._retrieval_node.run(state)
                state = state.model_copy(update=retrieval_updates)
                sample.retrieved_contexts = [
                    chunk.get("text", "") or chunk.get("child_text", "")
                    for chunk in state.retrieved_chunks
                ]

                # 2. Generate response
                result = self._response_service.generate(state)
                sample.response = result.get("draft_response", "")

            except Exception as exc:
                logger.warning("Pipeline failed for %s: %s", sample.id, exc)
                sample.response = ""
                sample.retrieved_contexts = []
            finally:
                progress.advance(sample.id)

        return samples

    def _init_pipeline_services(self) -> None:
        """Lazily initialise retriever and response service."""
        if self._retriever is None:
            from src.utils.nodes.retrieval import RetrievalNode
            from src.utils.vector_store import VectorStoreTools

            self._retriever = VectorStoreTools()
            self._retrieval_node = RetrievalNode(retriever=self._retriever)
        if self._response_service is None:
            from src.services.response_generation import ResponseGenerationService

            self._response_service = ResponseGenerationService()

    # -- RAGAS metric evaluation ---------------------------------------------

    def _evaluate_with_ragas(self, samples: list[RagasEvalSample]) -> Any:
        """Evaluate samples one by one to keep execution synchronous and observable."""
        import pandas as pd
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        metrics = self._select_metrics()
        evaluator_llm = self._get_evaluator_llm()
        evaluator_embeddings = self._get_evaluator_embeddings()
        frames: list[Any] = []
        progress = ProgressTracker(len(samples), "RAGAS")

        for sample in samples:
            logger.info("Evaluating RAGAS metrics for %s", sample.id)
            eval_dataset = EvaluationDataset(
                samples=[
                    SingleTurnSample(
                        user_input=sample.question,
                        retrieved_contexts=sample.retrieved_contexts,
                        response=sample.response,
                        reference=sample.ground_truth,
                        reference_contexts=sample.reference_contexts,
                    )
                ]
            )

            result = evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            )
            frame = result.to_pandas()
            frame.insert(0, "sample_id", sample.id)
            frames.append(frame)
            progress.advance(sample.id)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _before_openrouter_request(self, kind: str, model: str) -> None:
        with self._request_lock:
            now = time.perf_counter()
            min_delay = self.config.openrouter_min_interval_seconds + random.uniform(
                0.0, self.config.openrouter_jitter_seconds
            )
            wait_for = max((self._last_openrouter_request_at + min_delay) - now, 0.0)
            if wait_for > 0:
                logger.debug(
                    "Throttling OpenRouter %s request for %.2fs (model=%s)",
                    kind,
                    wait_for,
                    model,
                )
                time.sleep(wait_for)
            self._last_openrouter_request_at = time.perf_counter()

    def _should_retry_openrouter_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}:
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "cloudflare",
                "ddos",
                "rate limit",
                "too many requests",
                "temporarily unavailable",
                "timeout",
                "connection",
            )
        )

    def _extract_openrouter_usage(self, response: Any, headers: Any, kind: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        self._usage_summary["request_count"] += 1
        if kind == "llm":
            self._usage_summary["llm_request_count"] += 1
        else:
            self._usage_summary["embedding_request_count"] += 1
        self._usage_summary["prompt_tokens"] += prompt_tokens
        self._usage_summary["completion_tokens"] += completion_tokens
        self._usage_summary["total_tokens"] += total_tokens

        payload: dict[str, Any] = {}
        if hasattr(response, "model_dump"):
            try:
                payload = response.model_dump()
            except Exception:
                payload = {}

        candidate_costs: list[float] = []
        for source in (payload, payload.get("usage", {})):
            if isinstance(source, dict):
                for key, value in source.items():
                    if "cost" in str(key).lower() and isinstance(value, (int, float)):
                        candidate_costs.append(float(value))

        if headers is not None:
            for key, value in headers.items():
                if "cost" in str(key).lower():
                    try:
                        candidate_costs.append(float(value))
                    except (TypeError, ValueError):
                        continue

        if candidate_costs:
            reported_cost = max(candidate_costs)
            self._usage_summary["cost_usd_reported"] += reported_cost
            self._usage_summary["cost_fields"].append({"kind": kind, "cost_usd": reported_cost})

    def _wrap_openrouter_create(
        self,
        create_fn: Any,
        raw_create_fn: Any,
        kind: str,
        model: str,
    ) -> Any:
        def _wrapped_create(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, self.config.openrouter_max_retries + 1):
                self._before_openrouter_request(kind, model)
                try:
                    if raw_create_fn is not None:
                        raw_response = raw_create_fn(*args, **kwargs)
                        parsed = raw_response.parse()
                        self._extract_openrouter_usage(
                            parsed, getattr(raw_response, "headers", {}), kind
                        )
                        return parsed

                    response = create_fn(*args, **kwargs)
                    self._extract_openrouter_usage(response, {}, kind)
                    return response
                except Exception as exc:
                    last_exc = exc
                    if not self._should_retry_openrouter_error(exc):
                        raise
                    sleep_seconds = self.config.openrouter_backoff_base_seconds * (
                        2 ** (attempt - 1)
                    )
                    sleep_seconds += random.uniform(0.0, self.config.openrouter_jitter_seconds)
                    logger.warning(
                        "OpenRouter %s request failed (attempt %d/%d, model=%s): %s; "
                        "retrying in %.2fs",
                        kind,
                        attempt,
                        self.config.openrouter_max_retries,
                        model,
                        exc,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)

            assert last_exc is not None
            raise last_exc

        return _wrapped_create

    def _build_tracked_openrouter_client(self, model: str) -> Any:
        from openai import OpenAI

        from src.config import get_settings

        settings = get_settings()
        client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

        client.chat.completions.create = self._wrap_openrouter_create(
            client.chat.completions.create,
            getattr(getattr(client.chat.completions, "with_raw_response", None), "create", None),
            "llm",
            model,
        )
        client.embeddings.create = self._wrap_openrouter_create(
            client.embeddings.create,
            getattr(getattr(client.embeddings, "with_raw_response", None), "create", None),
            "embedding",
            model,
        )
        return client

    def _select_metrics(self) -> list[Any]:
        """Select metric instances based on the configured tier.

        Uses the *legacy* ``ragas.metrics`` imports (deprecated but still
        functional in v0.4.3).  These classes extend ``Metric`` which is
        the base type that ``evaluate()`` validates via ``isinstance()``.

        The newer ``ragas.metrics.collections`` classes do NOT inherit from
        ``Metric`` and therefore cause a ``TypeError`` when passed to
        ``evaluate()``.  LLM / embedding injection is handled by
        ``evaluate()`` itself — we do NOT pass them in the constructors.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            from ragas.metrics import (  # type: ignore[import-untyped]
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )

        # Tier 1 — Core RAG metrics (always run)
        # NOTE: Do NOT pass llm= / embeddings= here; evaluate() injects them.
        core: list[Any] = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

        if self.config.metrics_tier == "extended":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)

                from ragas.metrics import (  # type: ignore[import-untyped]
                    ContextEntityRecall,
                    FactualCorrectness,
                    NoiseSensitivity,
                    SemanticSimilarity,
                )

            core.extend(
                [
                    FactualCorrectness(),
                    SemanticSimilarity(),
                    ContextEntityRecall(),
                    NoiseSensitivity(),
                ]
            )

        return core

    # -- LLM / embeddings wrappers -------------------------------------------

    def _get_evaluator_llm(self) -> Any:
        """Get the RAGAS evaluator LLM via v0.4 llm_factory.

        Uses the OpenAI client configured with OpenRouter's base_url.
        ``llm_factory()`` auto-selects the best adapter for the provider.
        """
        if self._llm is not None:
            return self._llm

        from src.config import get_settings

        settings = get_settings()
        model = settings.RAGAS_LLM_MODEL or settings.LLM_MODEL

        try:
            from ragas.llms import llm_factory

            client = self._build_tracked_openrouter_client(model)
            self._llm = llm_factory(model=model, client=client)
        except Exception:
            # Fallback: wrap via LangchainLLMWrapper (deprecated but still works)
            logger.warning("llm_factory() failed; falling back to LangchainLLMWrapper")
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr
            from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]

            lc_llm = ChatOpenAI(
                model=model,
                api_key=SecretStr(settings.OPENAI_API_KEY or ""),
                base_url=settings.OPENAI_BASE_URL,
                temperature=0.0,
            )
            self._llm = LangchainLLMWrapper(lc_llm)

        return self._llm

    def _get_evaluator_embeddings(self) -> Any:
        """Get embeddings for RAGAS metrics using the legacy-compatible interface.

        ``ragas.evaluate()`` in this code path expects an embeddings object with
        ``embed_query`` / ``embed_documents`` methods. The newer
        ``ragas.embeddings.OpenAIEmbeddings`` provider exposes
        ``embed_text`` / ``embed_texts`` instead and will fail at runtime.
        """
        if self._embeddings is not None:
            return self._embeddings

        from src.config import get_settings

        settings = get_settings()
        model = settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL

        try:
            client = self._build_tracked_openrouter_client(model)

            class _TrackedEmbeddingAdapter:
                def __init__(self, adapter_client: Any, adapter_model: str):
                    self.client = adapter_client
                    self.model = adapter_model

                def embed_query(self, text: str) -> list[float]:
                    response = self.client.embeddings.create(input=text, model=self.model)
                    return list(response.data[0].embedding)

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    response = self.client.embeddings.create(input=texts, model=self.model)
                    return [list(item.embedding) for item in response.data]

                async def aembed_query(self, text: str) -> list[float]:
                    return self.embed_query(text)

                async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                    return self.embed_documents(texts)

                def set_run_config(self, run_config: Any) -> None:
                    self.run_config = run_config

            self._embeddings = _TrackedEmbeddingAdapter(client, model)
        except Exception as exc:
            logger.warning(
                "Tracked embedding adapter setup failed; falling back to adapter: %s",
                exc,
            )
            client = self._build_tracked_openrouter_client(model)

            class _OpenAIEmbeddingAdapter:
                def __init__(self, adapter_client: Any, adapter_model: str):
                    self.client = adapter_client
                    self.model = adapter_model

                def embed_query(self, text: str) -> list[float]:
                    response = self.client.embeddings.create(input=text, model=self.model)
                    return list(response.data[0].embedding)

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    response = self.client.embeddings.create(input=texts, model=self.model)
                    return [list(item.embedding) for item in response.data]

                async def aembed_query(self, text: str) -> list[float]:
                    return self.embed_query(text)

                async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                    return self.embed_documents(texts)

                def set_run_config(self, run_config: Any) -> None:
                    self.run_config = run_config

            self._embeddings = _OpenAIEmbeddingAdapter(client, model)

        return self._embeddings

    # -- report builder ------------------------------------------------------

    def _build_report(
        self,
        samples: list[RagasEvalSample],
        results: Any,
    ) -> dict[str, Any]:
        """Assemble the final JSON report with aggregate + per-sample scores."""
        import math

        from src.config import get_settings

        settings = get_settings()
        eval_date = self.config.evaluation_date

        # Extract per-sample scores from the RAGAS result object
        scores_df = results.to_pandas() if hasattr(results, "to_pandas") else results

        def _coerce_score(value: Any) -> float | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                numeric = float(value)
                if math.isnan(numeric):
                    return None
                return numeric
            return None

        excluded_columns = {
            "sample_id",
            "user_input",
            "response",
            "reference",
            "retrieved_contexts",
            "reference_contexts",
            "rubrics",
        }
        metric_columns: list[str] = []
        for col in scores_df.columns:
            if col in excluded_columns:
                continue
            series = scores_df[col]
            has_numeric_value = any(_coerce_score(value) is not None for value in series.tolist())
            if has_numeric_value:
                metric_columns.append(col)

        # Aggregate scores
        aggregate: dict[str, float] = {}
        for col in metric_columns:
            numeric_values = [
                coerced
                for coerced in (_coerce_score(value) for value in scores_df[col].tolist())
                if coerced is not None
            ]
            if numeric_values:
                aggregate[col] = round(sum(numeric_values) / len(numeric_values), 4)

        # Per-sample breakdown
        per_sample: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            row_scores: dict[str, float] = {}
            for col in metric_columns:
                coerced = _coerce_score(scores_df.iloc[idx][col])
                if coerced is not None:
                    row_scores[col] = round(coerced, 4)

            per_sample.append(
                {
                    "id": sample.id,
                    "question": sample.question,
                    "scores": row_scores,
                    "response_preview": sample.response[:200] if sample.response else "",
                    "retrieved_context_count": len(sample.retrieved_contexts),
                    "metadata": sample.metadata,
                }
            )

        return {
            "evaluation_type": "ragas_rag_eval",
            "evaluation_date": eval_date,
            "dataset_id": self.config.dataset_path.name,
            "sample_count": len(samples),
            "metrics_tier": self.config.metrics_tier,
            "aggregate_scores": aggregate,
            "per_sample": per_sample,
            "openrouter_usage": self._usage_summary,
            "config": {
                "evaluator_llm": settings.RAGAS_LLM_MODEL or settings.LLM_MODEL,
                "evaluator_embedding": settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL,
                "k_eval": self.config.k_eval,
                "retrieval_strategy": settings.RETRIEVAL_STRATEGY,
                "mode": self.config.mode,
                "openrouter_min_interval_seconds": self.config.openrouter_min_interval_seconds,
                "openrouter_jitter_seconds": self.config.openrouter_jitter_seconds,
                "openrouter_max_retries": self.config.openrouter_max_retries,
                "openrouter_backoff_base_seconds": self.config.openrouter_backoff_base_seconds,
            },
        }

    def _save_report(self, report: dict[str, Any]) -> None:
        """Persist the JSON report to disk (if an output path was configured)."""
        assert self.config.output_path is not None
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", self.config.output_path)
        logger.info("OpenRouter usage summary: %s", report.get("openrouter_usage"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS RAG evaluation on the PASsistant pipeline"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to RAGAS eval dataset (JSONL)",
    )
    parser.add_argument("--output", help="Path to write JSON report")
    parser.add_argument(
        "--k-eval",
        type=int,
        default=5,
        help="Retrieval depth (default: 5)",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "fixture"],
        default="live",
        help="Evaluation mode (default: live)",
    )
    parser.add_argument(
        "--fixture",
        help="Path to pre-computed responses JSON (fixture mode)",
    )
    parser.add_argument(
        "--metrics-tier",
        choices=["core", "extended"],
        default="core",
        help="Which metric tier to evaluate (default: core)",
    )
    parser.add_argument(
        "--evaluation-date",
        default=datetime.now(UTC).date().isoformat(),
        help="Evaluation date / report prefix in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--openrouter-min-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum delay between OpenRouter requests.",
    )
    parser.add_argument(
        "--openrouter-jitter-seconds",
        type=float,
        default=0.25,
        help="Random jitter added to OpenRouter request pacing.",
    )
    parser.add_argument(
        "--openrouter-max-retries",
        type=int,
        default=6,
        help="Maximum retries for transient OpenRouter / Cloudflare failures.",
    )
    parser.add_argument(
        "--openrouter-backoff-base-seconds",
        type=float,
        default=2.0,
        help="Base exponential backoff in seconds for transient OpenRouter failures.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for ``python -m src.eval.ragas_eval`` / ``ragas-eval``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    args = _parse_args()

    config = RagasEvalConfig(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output) if args.output else None,
        metrics_tier=args.metrics_tier,
        mode=args.mode,
        fixture_path=Path(args.fixture) if args.fixture else None,
        k_eval=args.k_eval,
        evaluation_date=args.evaluation_date,
        openrouter_min_interval_seconds=args.openrouter_min_interval_seconds,
        openrouter_jitter_seconds=args.openrouter_jitter_seconds,
        openrouter_max_retries=args.openrouter_max_retries,
        openrouter_backoff_base_seconds=args.openrouter_backoff_base_seconds,
    )

    evaluator = RagasEvaluator(config)
    report = asyncio.run(evaluator.run())

    # Summary output
    print("\n=== RAGAS Evaluation Summary ===")
    print(f"Samples evaluated: {report['sample_count']}")
    print(f"Metrics tier:      {report['metrics_tier']}")
    print("Aggregate scores:")
    for metric, score in report.get("aggregate_scores", {}).items():
        print(f"  {metric:40s}  {score:.4f}")

    if config.output_path:
        print(f"\nFull report: {config.output_path}")


if __name__ == "__main__":
    main()
