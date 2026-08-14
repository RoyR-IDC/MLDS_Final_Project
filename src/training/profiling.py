"""Lightweight training performance profiling helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA work when profiling a CUDA device."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_seconds(device: torch.device | None = None) -> "_TimedSeconds":
    """Return a context manager that measures elapsed seconds."""

    return _TimedSeconds(device=device)


@dataclass
class _TimedSeconds:
    device: torch.device | None = None
    elapsed_seconds: float = 0.0
    _start: float = 0.0

    def __enter__(self) -> "_TimedSeconds":
        if self.device is not None:
            synchronize_if_cuda(self.device)
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self.device is not None:
            synchronize_if_cuda(self.device)
        self.elapsed_seconds = perf_counter() - self._start
        return False


@dataclass
class PhaseProfiler:
    """Accumulate timing for one train or validation phase."""

    enabled: bool
    device: torch.device
    phase: str
    epoch: int | None
    warmup_batches: int = 0
    data_loading_seconds: float = 0.0
    transfer_seconds: float = 0.0
    compute_seconds: float = 0.0
    total_seconds: float = 0.0
    measured_batches: int = 0
    total_batches: int = 0
    _phase_start: float = field(default=0.0, init=False)

    def start(self) -> None:
        if not self.enabled:
            return
        synchronize_if_cuda(self.device)
        self._phase_start = perf_counter()

    def finish(self) -> None:
        if not self.enabled:
            return
        synchronize_if_cuda(self.device)
        self.total_seconds = perf_counter() - self._phase_start

    def add_data_loading(self, seconds: float) -> None:
        self._add_measured("data_loading_seconds", seconds)

    def add_transfer(self, seconds: float) -> None:
        self._add_measured("transfer_seconds", seconds)

    def add_compute(self, seconds: float) -> None:
        self._add_measured("compute_seconds", seconds)

    def count_batch(self) -> None:
        if not self.enabled:
            return
        self.total_batches += 1
        if self.total_batches > self.warmup_batches:
            self.measured_batches += 1

    def _add_measured(self, field_name: str, seconds: float) -> None:
        if not self.enabled or self.total_batches <= self.warmup_batches:
            return
        setattr(self, field_name, getattr(self, field_name) + seconds)

    def summary_row(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        """Return a CSV-friendly profile summary row."""

        if not self.enabled:
            return None
        return {
            **metadata,
            "epoch": self.epoch,
            "phase": self.phase,
            "total_batches": self.total_batches,
            "measured_batches": self.measured_batches,
            "data_loading_seconds": self.data_loading_seconds,
            "transfer_seconds": self.transfer_seconds,
            "compute_seconds": self.compute_seconds,
            "total_seconds": self.total_seconds,
            "avg_data_loading_seconds_per_batch": self.data_loading_seconds / max(1, self.measured_batches),
            "avg_transfer_seconds_per_batch": self.transfer_seconds / max(1, self.measured_batches),
            "avg_compute_seconds_per_batch": self.compute_seconds / max(1, self.measured_batches),
        }
