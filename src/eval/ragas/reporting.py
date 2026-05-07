from __future__ import annotations

import json
import logging
import math
from typing import Any

from src.config import get_settings
from src.eval.ragas.models import RagasEvalConfig, RagasEvalSample

logger = logging.getLogger(__name__)

REPORT_EXCLUDED_COLUMNS = {
    "sample_id",
    "user_input",
    "response",
    "reference",
    "retrieved_contexts",
    "reference_contexts",
    "rubrics",
}


def build_report(
    config: RagasEvalConfig,
    samples: list[RagasEvalSample],
    results: Any,
    usage_summary: dict[str, Any],
) -> dict[str, Any]:
    scores_df = results.to_pandas() if hasattr(results, "to_pandas") else results
    all_metric_columns = get_all_metric_columns(scores_df)
    metric_columns = get_metric_columns(scores_df, all_metric_columns)
    aggregate = build_aggregate_scores(scores_df, metric_columns)
    per_sample = build_per_sample_breakdown(samples, scores_df, all_metric_columns, metric_columns)
    settings = get_settings()

    return {
        "evaluation_type": "ragas_rag_eval",
        "evaluation_date": config.evaluation_date,
        "dataset_id": config.dataset_path.name,
        "sample_count": len(samples),
        "metrics_tier": config.metrics_tier,
        "aggregate_scores": aggregate,
        "per_sample": per_sample,
        "openrouter_usage": usage_summary,
        "config": {
            "evaluator_llm": settings.RAGAS_LLM_MODEL or settings.LLM_MODEL,
            "evaluator_embedding": settings.RAGAS_EMBEDDING_MODEL or settings.EMBEDDING_MODEL,
            "k_eval": config.k_eval,
            "retrieval_strategy": settings.RETRIEVAL_STRATEGY,
            "mode": config.mode,
            "openrouter_min_interval_seconds": config.openrouter_min_interval_seconds,
            "openrouter_jitter_seconds": config.openrouter_jitter_seconds,
            "openrouter_max_retries": config.openrouter_max_retries,
            "openrouter_backoff_base_seconds": config.openrouter_backoff_base_seconds,
        },
    }


def save_report(output_path: Any, report: dict[str, Any]) -> None:
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", output_path)
    logger.info("OpenRouter usage summary: %s", report.get("openrouter_usage"))


def get_all_metric_columns(scores_df: Any) -> list[str]:
    return [col for col in scores_df.columns if col not in REPORT_EXCLUDED_COLUMNS]


def get_metric_columns(scores_df: Any, all_metric_columns: list[str] | None = None) -> list[str]:
    candidate_columns = all_metric_columns or get_all_metric_columns(scores_df)
    metric_columns: list[str] = []
    for col in candidate_columns:
        series = scores_df[col]
        has_numeric_value = any(coerce_score(value) is not None for value in series.tolist())
        if has_numeric_value:
            metric_columns.append(col)
    return metric_columns


def build_aggregate_scores(scores_df: Any, metric_columns: list[str]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for col in metric_columns:
        numeric_values = [
            coerced for coerced in (coerce_score(value) for value in scores_df[col].tolist()) if coerced is not None
        ]
        if numeric_values:
            aggregate[col] = round(sum(numeric_values) / len(numeric_values), 4)
    return aggregate


def build_per_sample_breakdown(
    samples: list[RagasEvalSample],
    scores_df: Any,
    all_metric_columns: list[str],
    metric_columns: list[str],
) -> list[dict[str, Any]]:
    per_sample: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        row_scores: dict[str, float] = {}
        row_metric_errors: dict[str, str] = {}
        for col in metric_columns:
            coerced = coerce_score(scores_df.iloc[idx][col])
            if coerced is not None:
                row_scores[col] = round(coerced, 4)

        for col in all_metric_columns:
            if col in row_scores:
                continue
            coerced = coerce_score(scores_df.iloc[idx][col])
            if coerced is None:
                row_metric_errors[col] = get_metric_error_reason(col)

        per_sample.append(
            {
                "id": sample.id,
                "question": sample.question,
                "scores": row_scores,
                "metric_errors": row_metric_errors,
                "response_preview": sample.response[:200] if sample.response else "",
                "retrieved_context_count": len(sample.retrieved_contexts),
                "metadata": sample.metadata,
            }
        )
    return per_sample


def get_metric_error_reason(metric_name: str) -> str:
    metric_name = metric_name.lower()
    if metric_name == "faithfulness":
        return (
            "Metric returned null/NaN from RAGAS, typically because no answer statements "
            "were extracted or the faithfulness judgment could not be completed."
        )
    if metric_name == "context_recall":
        return (
            "Metric returned null/NaN from RAGAS, typically because the evaluator output "
            "could not be parsed into the expected attribution structure."
        )
    if metric_name == "answer_relevancy":
        return (
            "Metric returned null/NaN from RAGAS, typically because the evaluator output "
            "could not be parsed or the metric could not complete successfully."
        )
    if metric_name.startswith("context_precision"):
        return (
            "Metric returned null/NaN from RAGAS, typically because the evaluator output "
            "could not be parsed or the metric could not complete successfully."
        )
    return "Metric returned null/NaN from RAGAS."


def coerce_score(value: Any) -> float | None:
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
