"""RAGAS-based RAG evaluation for the PASsistant pipeline (RAGAS v0.4.3+).

Supports two evaluation modes:
- **live**: runs the full RAG pipeline (retrieval + generation) for each sample
- **fixture**: loads pre-computed responses from a JSON file

Usage:
    python -m src.eval.ragas_eval \\
        --dataset tests/fixtures/ragas_eval_dataset.jsonl \\
        --mode live \\
        --output reports/ragas_eval_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
    batch_size: int = 5
    evaluation_date: str | None = None  # optional YYYY-MM-DD override


@dataclass
class RagasEvalSample:
    """Single evaluation sample with all pipeline outputs."""

    id: str
    user_input: str
    reference: str
    reference_contexts: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"id", "user_input", "reference"}


def load_dataset(path: Path) -> list[RagasEvalSample]:
    """Load and validate a JSONL evaluation dataset.

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

            missing = _REQUIRED_FIELDS - set(row.keys())
            if missing:
                raise ValueError(
                    f"Line {line_no} missing required field(s): {', '.join(sorted(missing))}"
                )

            samples.append(
                RagasEvalSample(
                    id=row["id"],
                    user_input=row["user_input"],
                    reference=row["reference"],
                    reference_contexts=row.get("reference_contexts", []),
                    metadata=row.get("metadata", {}),
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

    # -- public entry point --------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Execute the full evaluation pipeline and return a JSON report."""
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
        """Fill samples from a pre-computed fixture file."""
        if self.config.fixture_path is None:
            raise ValueError("--fixture path is required when mode=fixture")
        fixtures = load_fixtures(self.config.fixture_path)
        for sample in samples:
            fixture = fixtures.get(sample.id, {})
            sample.response = fixture.get("response", "")
            sample.retrieved_contexts = fixture.get("retrieved_contexts", [])
        return samples

    async def _run_live_pipeline(self, samples: list[RagasEvalSample]) -> list[RagasEvalSample]:
        """Run the real RAG pipeline for each sample."""
        self._init_pipeline_services()

        for sample in samples:
            try:
                from langchain_core.messages import HumanMessage

                from src.utils.state import AgentState

                state = AgentState(
                    messages=[HumanMessage(content=sample.user_input)],
                    requires_retrieval=True,
                    retrieval_query=sample.user_input,
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

        return samples

    def _init_pipeline_services(self) -> None:
        """Lazily initialise retriever and response service."""
        if self._retriever is None:
            from src.utils.vector_store import VectorStoreTools
            from src.utils.nodes.retrieval import RetrievalNode

            self._retriever = VectorStoreTools()
            self._retrieval_node = RetrievalNode(retriever=self._retriever)
        if self._response_service is None:
            from src.services.response_generation import ResponseGenerationService

            self._response_service = ResponseGenerationService()

    # -- RAGAS metric evaluation ---------------------------------------------

    def _evaluate_with_ragas(self, samples: list[RagasEvalSample]) -> Any:
        """Convert samples to a RAGAS EvaluationDataset and run evaluate().

        Uses the legacy ``evaluate()`` API which expects ``Metric`` subclasses.
        The ``llm`` and ``embeddings`` parameters are passed to ``evaluate()``
        which injects them into every metric that needs them.
        """
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        ragas_samples = [
            SingleTurnSample(
                user_input=s.user_input,
                retrieved_contexts=s.retrieved_contexts,
                response=s.response,
                reference=s.reference,
                reference_contexts=s.reference_contexts,
            )
            for s in samples
        ]

        eval_dataset = EvaluationDataset(samples=ragas_samples)
        metrics = self._select_metrics()
        evaluator_llm = self._get_evaluator_llm()
        evaluator_embeddings = self._get_evaluator_embeddings()

        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        return results

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
            from openai import OpenAI
            from ragas.llms import llm_factory

            client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )
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
        """Get embeddings for RAGAS SemanticSimilarity / AnswerRelevancy.

        Uses the modern ``ragas.embeddings.OpenAIEmbeddings`` class directly.
        Falls back to wrapping ``langchain_openai.OpenAIEmbeddings`` via
        ``LangchainEmbeddingsWrapper`` if the RAGAS native class is unavailable.
        """
        if self._embeddings is not None:
            return self._embeddings

        from src.config import get_settings

        settings = get_settings()
        model = settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL

        try:
            from openai import OpenAI
            from ragas.embeddings import (
                OpenAIEmbeddings as RagasOpenAIEmbeddings,  # type: ignore[import-untyped]
            )

            client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )
            self._embeddings = RagasOpenAIEmbeddings(client=client, model=model)
        except Exception:
            logger.warning(
                "ragas.embeddings.OpenAIEmbeddings failed; falling back to LangchainEmbeddingsWrapper"
            )
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr
            from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]

            lc_emb = OpenAIEmbeddings(
                model=model,
                api_key=SecretStr(settings.OPENAI_API_KEY or ""),
                base_url=settings.OPENAI_BASE_URL,
            )
            self._embeddings = LangchainEmbeddingsWrapper(lc_emb)

        return self._embeddings

    # -- report builder ------------------------------------------------------

    def _build_report(
        self,
        samples: list[RagasEvalSample],
        results: Any,
    ) -> dict[str, Any]:
        """Assemble the final JSON report with aggregate + per-sample scores."""
        from src.config import get_settings

        settings = get_settings()
        eval_date = self.config.evaluation_date or datetime.now(UTC).strftime("%Y-%m-%d")

        # Extract per-sample scores from the RAGAS result object
        scores_df = results.to_pandas()  # DataFrame indexed like the input
        metric_columns = [
            col for col in scores_df.columns if col not in {"user_input", "response", "reference"}
        ]

        # Aggregate scores
        aggregate: dict[str, float] = {}
        for col in metric_columns:
            values = scores_df[col].dropna()
            if len(values) > 0:
                aggregate[col] = round(float(values.mean()), 4)

        # Per-sample breakdown
        per_sample: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            row_scores: dict[str, float] = {}
            for col in metric_columns:
                val = scores_df.iloc[idx][col]
                if val is not None and str(val) != "nan":
                    row_scores[col] = round(float(val), 4)

            per_sample.append(
                {
                    "id": sample.id,
                    "user_input": sample.user_input,
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
            "config": {
                "evaluator_llm": settings.RAGAS_LLM_MODEL or settings.LLM_MODEL,
                "evaluator_embedding": settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL,
                "k_eval": self.config.k_eval,
                "retrieval_strategy": settings.RETRIEVAL_STRATEGY,
                "mode": self.config.mode,
            },
        }

    def _save_report(self, report: dict[str, Any]) -> None:
        """Persist the JSON report to disk (if an output path was configured)."""
        if self.config.output_path is None:
            logger.info(
                "No --output path specified; skipping report write. Aggregate scores: %s",
                report.get("aggregate_scores"),
            )
            return

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", self.config.output_path)


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
        default=None,
        help="Optional YYYY-MM-DD override for the report date",
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
