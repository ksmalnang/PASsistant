from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RagasEvalConfig:
    """Configuration for a RAGAS evaluation run."""

    dataset_path: Path
    output_path: Path | None = None
    metrics_tier: str = "core"
    mode: str = "live"
    fixture_path: Path | None = None
    k_eval: int = 10
    batch_size: int = 1
    evaluation_date: str = field(default_factory=lambda: datetime.now(UTC).date().isoformat())
    openrouter_min_interval_seconds: float = 0.3
    openrouter_jitter_seconds: float = 0.15
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
