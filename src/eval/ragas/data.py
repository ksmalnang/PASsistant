from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.eval.ragas.models import RagasEvalSample

logger = logging.getLogger(__name__)

DATASET_REQUIRED_FIELDS = {"id", "question", "contexts", "answer", "ground_truth", "metadata"}
METADATA_REQUIRED_FIELDS = {"source_file", "difficulty", "reasoning_type", "noise_level"}
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}
ALLOWED_REASONING_TYPE = {"single-hop", "multi-hop", "comparison", "inference"}
ALLOWED_NOISE_LEVEL = {"low", "medium", "high"}


def load_dataset(path: Path) -> list[RagasEvalSample]:
    """Load and validate a JSONL evaluation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[RagasEvalSample] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            samples.append(_parse_dataset_row(row, line_no))

    if not samples:
        raise ValueError(f"Dataset is empty: {path}")

    logger.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


def load_fixtures(path: Path) -> dict[str, dict[str, Any]]:
    """Load pre-computed pipeline responses for fixture mode."""
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Fixture file must contain a JSON object mapping sample IDs to outputs")

    logger.info("Loaded fixtures for %d samples from %s", len(data), path)
    return data


def _parse_dataset_row(row: Any, line_no: int) -> RagasEvalSample:
    if not isinstance(row, dict):
        raise ValueError(f"Line {line_no}: row must be a JSON object")

    missing = DATASET_REQUIRED_FIELDS - set(row.keys())
    if missing:
        raise ValueError(f"Line {line_no} missing required field(s): {', '.join(sorted(missing))}")

    sample_id = _require_non_empty_string(row.get("id"), "id", line_no)
    question = _require_non_empty_string(row.get("question"), "question", line_no)
    answer = _require_non_empty_string(row.get("answer"), "answer", line_no)
    ground_truth = _require_non_empty_string(row.get("ground_truth"), "ground_truth", line_no)
    provided_contexts, reference_contexts = _parse_contexts(row.get("contexts"), line_no)
    metadata = _parse_metadata(row.get("metadata"), line_no)

    return RagasEvalSample(
        id=sample_id,
        question=question,
        ground_truth=ground_truth,
        provided_contexts=provided_contexts,
        reference_contexts=reference_contexts,
        response=answer,
        metadata=metadata,
    )


def _parse_metadata(metadata: Any, line_no: int) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"Line {line_no}: metadata must be an object")

    missing = METADATA_REQUIRED_FIELDS - set(metadata.keys())
    if missing:
        raise ValueError(
            f"Line {line_no}: metadata missing required field(s): {', '.join(sorted(missing))}"
        )

    parsed_metadata = {
        "source_file": _require_non_empty_string(metadata.get("source_file"), "metadata.source_file", line_no),
        "difficulty": _require_enum_value(
            metadata.get("difficulty"),
            "metadata.difficulty",
            ALLOWED_DIFFICULTY,
            line_no,
        ),
        "reasoning_type": _require_enum_value(
            metadata.get("reasoning_type"),
            "metadata.reasoning_type",
            ALLOWED_REASONING_TYPE,
            line_no,
        ),
        "noise_level": _require_enum_value(
            metadata.get("noise_level"),
            "metadata.noise_level",
            ALLOWED_NOISE_LEVEL,
            line_no,
        ),
    }

    for key, value in metadata.items():
        if key not in parsed_metadata:
            parsed_metadata[key] = value

    return parsed_metadata


def _require_non_empty_string(value: Any, field_name: str, line_no: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_no}: {field_name} must be a non-empty string")
    return value


def _require_enum_value(value: Any, field_name: str, allowed: set[str], line_no: int) -> str:
    if not isinstance(value, str) or value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Line {line_no}: {field_name} must be one of: {allowed_values}")
    return value


def _parse_contexts(contexts: Any, line_no: int) -> tuple[list[str], list[str]]:
    if not isinstance(contexts, list):
        raise ValueError(f"Line {line_no}: contexts must be a list")

    provided_contexts: list[str] = []
    reference_contexts: list[str] = []
    for idx, context in enumerate(contexts, start=1):
        if not isinstance(context, dict):
            raise ValueError(f"Line {line_no}: contexts[{idx}] must be an object")

        text = context.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Line {line_no}: contexts[{idx}].text must be a non-empty string")

        is_relevant = context.get("is_relevant")
        if is_relevant is not None and not isinstance(is_relevant, bool):
            raise ValueError(f"Line {line_no}: contexts[{idx}].is_relevant must be a boolean")

        provided_contexts.append(text)
        if context.get("is_relevant", True):
            reference_contexts.append(text)

    if not provided_contexts:
        raise ValueError(f"Line {line_no}: contexts must contain at least one item")

    return provided_contexts, reference_contexts
