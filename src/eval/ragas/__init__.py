"""RAGAS-based RAG evaluation for the PASsistant pipeline."""

from __future__ import annotations

from src.eval.ragas.cli import _parse_args, main
from src.eval.ragas.data import load_dataset, load_fixtures
from src.eval.ragas.evaluator import RagasEvaluator
from src.eval.ragas.models import ProgressTracker, RagasEvalConfig, RagasEvalSample

__all__ = [
    "ProgressTracker",
    "RagasEvalConfig",
    "RagasEvalSample",
    "RagasEvaluator",
    "load_dataset",
    "load_fixtures",
    "_parse_args",
    "main",
]
