# ==============================
# File: linalg/core/mode.py
# ==============================
"""Execution modes for the linalg engine."""

from __future__ import annotations

import enum

__all__ = ["ExecutionMode"]


class ExecutionMode(enum.Enum):
    """Supported execution modes.

    * **FAST** — default; skip tracing, minimal overhead.
    * **EDUCATIONAL** — record every intermediate step for replay.
    * **DEBUG** — same as EDUCATIONAL but also prints live diagnostics.
    """

    FAST = "fast"
    EDUCATIONAL = "educational"
    DEBUG = "debug"

    @classmethod
    def from_string(cls, value: str) -> ExecutionMode:
        """Parse a mode string (case-insensitive)."""
        normalised = value.strip().lower()
        for member in cls:
            if member.value == normalised:
                return member
        from mllense.math.linalg.exceptions import InvalidModeError

        raise InvalidModeError(value, tuple(m.value for m in cls))
