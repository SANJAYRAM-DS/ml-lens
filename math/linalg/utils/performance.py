# ==============================
# File: linalg/utils/performance.py
# ==============================
"""Performance benchmarking utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List

__all__ = ["BenchmarkResult", "benchmark"]


@dataclass
class BenchmarkResult:
    """Container for benchmark measurements.

    Attributes:
        name: Human-readable benchmark name.
        iterations: Number of iterations executed.
        total_seconds: Total elapsed wall-clock time.
        mean_seconds: Mean time per iteration.
        min_seconds: Fastest iteration.
        max_seconds: Slowest iteration.
    """

    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    timings: List[float] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(name={self.name!r}, "
            f"iters={self.iterations}, "
            f"mean={self.mean_seconds:.6f}s, "
            f"min={self.min_seconds:.6f}s, "
            f"max={self.max_seconds:.6f}s)"
        )


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    iterations: int = 10,
    warmup: int = 1,
    name: str = "",
    **kwargs: Any,
) -> BenchmarkResult:
    """Benchmark a callable over multiple iterations.

    Args:
        func: The callable to benchmark.
        *args: Positional arguments for *func*.
        iterations: Number of timed iterations.
        warmup: Number of untimed warmup calls.
        name: Human-readable label.
        **kwargs: Keyword arguments for *func*.

    Returns:
        A :class:`BenchmarkResult` with timing statistics.
    """
    label = name or getattr(func, "__qualname__", str(func))

    # warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    timings: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    total = sum(timings)
    return BenchmarkResult(
        name=label,
        iterations=iterations,
        total_seconds=total,
        mean_seconds=total / iterations,
        min_seconds=min(timings),
        max_seconds=max(timings),
        timings=timings,
    )
