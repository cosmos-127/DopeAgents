"""Real-time progress tracking for research workflows."""

from __future__ import annotations

import time

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepProgress:
    """Progress info for a single step."""

    step_name: str
    status: StepStatus = StepStatus.PENDING
    started_at: float | None = None
    completed_at: float | None = None
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return round(self.completed_at - self.started_at, 2)
        if self.started_at:
            return round(time.monotonic() - self.started_at, 2)
        return None


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(
        self,
        step: str,
        status: StepStatus,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass
class ResearchProgress:
    """Tracks overall research progress across all steps.

    Usage::

        progress = ResearchProgress(callback=my_callback)
        progress.start_step("expand_query", "Expanding research query...")
        progress.complete_step("expand_query", "Generated 4 queries", {"count": 4})
    """

    steps: list[StepProgress] = field(default_factory=list)
    callback: ProgressCallback | None = None
    _step_map: dict[str, StepProgress] = field(default_factory=dict, repr=False)

    STEP_ORDER: list[str] = field(
        default_factory=lambda: [
            "load_context",
            "expand_query",
            "real_search",
            "extract_content",
            "chunk_and_rank",
            "score_sources",
            "extract_claims",
            "verify_claims",
            "cross_reference",
            "synthesize",
            "calculate_confidence",
            "evaluate",
            "generate_report",
            "save_session",
            "refine",
        ]
    )

    def __post_init__(self) -> None:
        if not self.steps:
            for step_name in self.STEP_ORDER:
                sp = StepProgress(step_name=step_name)
                self.steps.append(sp)
                self._step_map[step_name] = sp

    @property
    def current_step(self) -> str | None:
        for step in self.steps:
            if step.status == StepStatus.RUNNING:
                return step.step_name
        return None

    @property
    def completion_fraction(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(
            1 for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )
        return completed / len(self.steps)

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds or 0.0 for s in self.steps)

    def start_step(self, step_name: str, message: str = "") -> None:
        sp = self._step_map.get(step_name)
        if sp is None:
            sp = StepProgress(step_name=step_name)
            self.steps.append(sp)
            self._step_map[step_name] = sp

        sp.status = StepStatus.RUNNING
        sp.started_at = time.monotonic()
        sp.message = message

        if self.callback:
            self.callback(step_name, StepStatus.RUNNING, message)

    def complete_step(
        self, step_name: str, message: str = "", detail: dict[str, Any] | None = None
    ) -> None:
        sp = self._step_map[step_name]
        sp.status = StepStatus.COMPLETED
        sp.completed_at = time.monotonic()
        sp.message = message
        if detail:
            sp.detail = detail

        if self.callback:
            self.callback(step_name, StepStatus.COMPLETED, message, detail)

    def fail_step(self, step_name: str, error: str) -> None:
        sp = self._step_map[step_name]
        sp.status = StepStatus.FAILED
        sp.completed_at = time.monotonic()
        sp.error = error

        if self.callback:
            self.callback(step_name, StepStatus.FAILED, error)

    def summary(self) -> dict[str, Any]:
        return {
            "completion": f"{self.completion_fraction:.0%}",
            "current_step": self.current_step,
            "total_duration_s": round(self.total_duration, 1),
            "steps": [
                {
                    "name": s.step_name,
                    "status": s.status.value,
                    "duration_s": s.duration_seconds,
                    "message": s.message,
                }
                for s in self.steps
            ],
        }

    def render_progress_bar(self, width: int = 40) -> str:
        """Render a text-based progress bar."""
        filled = int(self.completion_fraction * width)
        bar = "█" * filled + "░" * (width - filled)
        pct = f"{self.completion_fraction:.0%}"
        current = self.current_step or "done"
        return f"[{bar}] {pct} — {current}"


def console_progress_callback(
    step: str,
    status: StepStatus,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Print progress to console with icons."""
    icons = {
        StepStatus.RUNNING: "🔄",
        StepStatus.COMPLETED: "✅",
        StepStatus.FAILED: "❌",
        StepStatus.SKIPPED: "⏭️",
    }
    icon = icons.get(status, "⬜")
    detail_str = ""
    if detail:
        detail_str = " | " + ", ".join(f"{k}={v}" for k, v in detail.items())
    print(f"  {icon} {step}: {message}{detail_str}")
