# ==============================
# File: linalg/utils/logging.py
# ==============================
"""Structured logging setup for the linalg engine."""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_CONFIGURED = False


def _configure_root_logger() -> None:
    """Lazy one-time configuration of the linalg logger hierarchy."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger("mllense.math.linalg")
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``mllense.math.linalg`` namespace.

    Usage::

        from mllense.math.linalg.utils.logging import get_logger
        log = get_logger(__name__)
        log.info("Starting computation")
    """
    _configure_root_logger()
    return logging.getLogger(f"mllense.math.linalg.{name}")
