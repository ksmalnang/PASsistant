"""Tests for the RAGAS evaluation module (src.eval.ragas_eval)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.eval.ragas_eval import (
    RagasEvalConfig,
    RagasEvalSample,
    RagasEvaluator,
    load_dataset,
    load_fixtures,
)

# ── fixtures directory ───────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _can_import_ragas() -> bool:
    """Check whether RAGAS is importable in this environment."""
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def _can_import_pandas() -> bool:
    try:
        import pandas  # noqa: F401

        return True
    except ImportError:
        return False


# ===========================================================================
# Phase 5.1 — Dataset loader tests
# ===========================================================================


class TestLoadDataset:
    """Unit tests for load_dataset()."""

    def test_valid_dataset(self, tmp_path: Path) -> None:
        """Valid JSONL loads successfully."""
        ds = tmp_path / "valid.jsonl"
        ds.write_text(
            textwrap.dedent("""\
            {"id": "q1", "user_input": "Hello?", "reference": "Hi!"}
            {"id": "q2", "user_input": "What?", "reference": "That.", "reference_contexts": ["ctx"]}
            """),
            encoding="utf-8",
        )
        samples = load_dataset(ds)
        assert len(samples) == 2
        assert samples[0].id == "q1"
        assert samples[1].reference_contexts == ["ctx"]

    def test_missing_required_field(self, tmp_path: Path) -> None:
        """Missing required field raises ValueError."""
        ds = tmp_path / "bad.jsonl"
        ds.write_text('{"id": "q1", "user_input": "Hello?"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field"):
            load_dataset(ds)

    def test_empty_dataset(self, tmp_path: Path) -> None:
        """Empty file raises ValueError."""
        ds = tmp_path / "empty.jsonl"
        ds.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_dataset(ds)

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Malformed JSON line raises ValueError."""
        ds = tmp_path / "malformed.jsonl"
        ds.write_text("{bad json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_dataset(ds)

    def test_file_not_found(self) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/does/not/exist.jsonl"))

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines in JSONL should be silently skipped."""
        ds = tmp_path / "blanks.jsonl"
        ds.write_text(
            '\n{"id": "q1", "user_input": "Hey", "reference": "Hi"}\n\n',
            encoding="utf-8",
        )
        samples = load_dataset(ds)
        assert len(samples) == 1

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        """Metadata dict should be preserved on the sample."""
        ds = tmp_path / "meta.jsonl"
        ds.write_text(
            '{"id": "q1", "user_input": "Q", "reference": "A", "metadata": {"cat": "test"}}\n',
            encoding="utf-8",
        )
        samples = load_dataset(ds)
        assert samples[0].metadata == {"cat": "test"}

    def test_real_fixture_dataset(self) -> None:
        """The actual evaluation dataset loads without errors."""
        ds_path = FIXTURES_DIR / "ragas_eval_dataset.jsonl"
        if not ds_path.exists():
            pytest.skip("Evaluation dataset not present")
        samples = load_dataset(ds_path)
        assert len(samples) >= 1


# ===========================================================================
# Phase 5.2 — Metric selection tests
# ===========================================================================


class TestMetricSelection:
    """Unit tests for RagasEvaluator._select_metrics()."""

    def _make_evaluator(self, tier: str = "core") -> RagasEvaluator:
        config = RagasEvalConfig(
            dataset_path=Path("dummy.jsonl"),
            metrics_tier=tier,
        )
        return RagasEvaluator(config)

    @pytest.mark.skipif(not _can_import_ragas(), reason="ragas not installed")
    def test_core_tier_has_four_metrics(self) -> None:
        evaluator = self._make_evaluator("core")
        metrics = evaluator._select_metrics()
        assert len(metrics) == 4

    @pytest.mark.skipif(not _can_import_ragas(), reason="ragas not installed")
    def test_extended_tier_has_eight_metrics(self) -> None:
        evaluator = self._make_evaluator("extended")
        metrics = evaluator._select_metrics()
        assert len(metrics) == 8

    @pytest.mark.skipif(not _can_import_ragas(), reason="ragas not installed")
    def test_core_metrics_are_ragas_metric_subclass(self) -> None:
        """Metrics must be Metric subclasses for evaluate() compatibility."""
        from ragas.metrics.base import Metric

        evaluator = self._make_evaluator("core")
        metrics = evaluator._select_metrics()
        for m in metrics:
            assert isinstance(m, Metric), f"{type(m).__name__} is not a Metric subclass"


# ===========================================================================
# Phase 5.3 — Integration test with mocked RAGAS
# ===========================================================================


class TestReportBuilder:
    """Verify report structure from a mocked evaluation run."""

    def _make_report(
        self,
        samples: list[RagasEvalSample] | None = None,
        tier: str = "core",
    ) -> dict[str, Any]:
        """Build a report from mocked RAGAS results."""
        pytest.importorskip("pandas")
        import pandas as pd

        config = RagasEvalConfig(
            dataset_path=Path("test.jsonl"),
            output_path=None,
            metrics_tier=tier,
            evaluation_date="2026-05-04",
        )
        evaluator = RagasEvaluator(config)

        if samples is None:
            samples = [
                RagasEvalSample(
                    id="q1",
                    user_input="Hello?",
                    reference="Hi!",
                    response="Hi there!",
                    retrieved_contexts=["ctx1"],
                    metadata={"category": "test"},
                ),
                RagasEvalSample(
                    id="q2",
                    user_input="What?",
                    reference="That.",
                    response="This.",
                    retrieved_contexts=["ctx2"],
                ),
            ]

        # Mock RAGAS result object
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = pd.DataFrame(
            {
                "faithfulness": [0.9, 0.8],
                "answer_relevancy": [0.85, 0.75],
                "context_precision_with_reference": [1.0, 0.9],
                "context_recall": [0.7, 0.6],
            }
        )

        return evaluator._build_report(samples, mock_result)

    @pytest.mark.skipif(not _can_import_pandas(), reason="pandas not installed")
    def test_report_has_required_keys(self) -> None:
        report = self._make_report()
        assert report["evaluation_type"] == "ragas_rag_eval"
        assert report["evaluation_date"] == "2026-05-04"
        assert report["sample_count"] == 2
        assert "aggregate_scores" in report
        assert "per_sample" in report
        assert "config" in report

    @pytest.mark.skipif(not _can_import_pandas(), reason="pandas not installed")
    def test_aggregate_scores_in_range(self) -> None:
        report = self._make_report()
        for metric, score in report["aggregate_scores"].items():
            assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

    @pytest.mark.skipif(not _can_import_pandas(), reason="pandas not installed")
    def test_per_sample_scores_present(self) -> None:
        report = self._make_report()
        assert len(report["per_sample"]) == 2
        for entry in report["per_sample"]:
            assert "id" in entry
            assert "scores" in entry
            for metric, score in entry["scores"].items():
                assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

    @pytest.mark.skipif(not _can_import_pandas(), reason="pandas not installed")
    def test_metadata_in_per_sample(self) -> None:
        report = self._make_report()
        assert report["per_sample"][0]["metadata"] == {"category": "test"}

    def test_report_saves_to_file(self, tmp_path: Path) -> None:
        """When output_path is set, the report should be saved as JSON."""
        output = tmp_path / "report.json"
        config = RagasEvalConfig(
            dataset_path=Path("test.jsonl"),
            output_path=output,
            evaluation_date="2026-05-04",
        )
        evaluator = RagasEvaluator(config)

        report = {"evaluation_type": "ragas_rag_eval", "sample_count": 0}
        evaluator._save_report(report)

        assert output.exists()
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["evaluation_type"] == "ragas_rag_eval"


# ===========================================================================
# Phase 5.4 — Fixture mode tests
# ===========================================================================


class TestFixtureMode:
    """Verify fixture loading and application to samples."""

    def test_load_fixtures(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "fixtures.json"
        fixture_data = {
            "q1": {
                "response": "The answer is 42.",
                "retrieved_contexts": ["context chunk 1"],
            },
        }
        fixture_path.write_text(json.dumps(fixture_data), encoding="utf-8")
        loaded = load_fixtures(fixture_path)
        assert "q1" in loaded
        assert loaded["q1"]["response"] == "The answer is 42."

    def test_fixture_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_fixtures(Path("/no/such/file.json"))

    def test_invalid_fixture_format(self, tmp_path: Path) -> None:
        fixture_path = tmp_path / "bad.json"
        fixture_path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_fixtures(fixture_path)

    def test_apply_fixtures_populates_samples(self, tmp_path: Path) -> None:
        """Fixture mode fills sample.response and sample.retrieved_contexts."""
        fixture_path = tmp_path / "f.json"
        fixture_path.write_text(
            json.dumps(
                {
                    "q1": {
                        "response": "Answer A",
                        "retrieved_contexts": ["ctx A"],
                    },
                    "q2": {
                        "response": "Answer B",
                        "retrieved_contexts": ["ctx B1", "ctx B2"],
                    },
                }
            ),
            encoding="utf-8",
        )

        config = RagasEvalConfig(
            dataset_path=Path("dummy.jsonl"),
            mode="fixture",
            fixture_path=fixture_path,
        )
        evaluator = RagasEvaluator(config)

        samples = [
            RagasEvalSample(id="q1", user_input="Q1?", reference="R1"),
            RagasEvalSample(id="q2", user_input="Q2?", reference="R2"),
        ]

        updated = evaluator._apply_fixtures(samples)
        assert updated[0].response == "Answer A"
        assert updated[1].retrieved_contexts == ["ctx B1", "ctx B2"]

    def test_fixture_mode_missing_path_raises(self) -> None:
        config = RagasEvalConfig(
            dataset_path=Path("dummy.jsonl"),
            mode="fixture",
            fixture_path=None,
        )
        evaluator = RagasEvaluator(config)
        with pytest.raises(ValueError, match="--fixture path is required"):
            evaluator._apply_fixtures([])
