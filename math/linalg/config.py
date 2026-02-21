# ==============================
# File: linalg/config.py
# ==============================
"""Global configuration for the linalg engine.

Provides a thread-safe singleton that stores runtime defaults.
This is the *only* mutable global state in the library.
"""

from __future__ import annotations

import threading
from typing import Any

__all__ = ["GlobalConfig", "get_config"]


class GlobalConfig:
    """Thread-safe global configuration container.

    Attributes:
        default_backend: Name of the default backend (``"numpy"`` | ``"python"``).
        default_mode: Execution mode (``"fast"`` | ``"educational"`` | ``"debug"``).
        trace_enabled: Whether trace recording is turned on globally.
        auto_algorithm_selection: If ``True``, the registry picks the best algorithm
            automatically based on input size and backend.
    """

    _instance: GlobalConfig | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> GlobalConfig:
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._default_backend = "numpy"
                inst._default_mode = "fast"
                inst._trace_enabled = False
                inst._auto_algorithm_selection = True
                cls._instance = inst
            return cls._instance

    # -- default_backend -------------------------------------------------- #
    @property
    def default_backend(self) -> str:
        return self._default_backend  # type: ignore[attr-defined]

    @default_backend.setter
    def default_backend(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            from mllense.math.linalg.exceptions import InvalidBackendError

            raise InvalidBackendError(str(value))
        self._default_backend = value.strip().lower()  # type: ignore[attr-defined]

    # -- default_mode ----------------------------------------------------- #
    @property
    def default_mode(self) -> str:
        return self._default_mode  # type: ignore[attr-defined]

    @default_mode.setter
    def default_mode(self, value: str) -> None:
        from mllense.math.linalg.core.mode import ExecutionMode

        valid = {m.value for m in ExecutionMode}
        if value not in valid:
            from mllense.math.linalg.exceptions import InvalidModeError

            raise InvalidModeError(value, tuple(valid))
        self._default_mode = value  # type: ignore[attr-defined]

    # -- trace_enabled ---------------------------------------------------- #
    @property
    def trace_enabled(self) -> bool:
        return self._trace_enabled  # type: ignore[attr-defined]

    @trace_enabled.setter
    def trace_enabled(self, value: bool) -> None:
        self._trace_enabled = bool(value)  # type: ignore[attr-defined]

    # -- auto_algorithm_selection ----------------------------------------- #
    @property
    def auto_algorithm_selection(self) -> bool:
        return self._auto_algorithm_selection  # type: ignore[attr-defined]

    @auto_algorithm_selection.setter
    def auto_algorithm_selection(self, value: bool) -> None:
        self._auto_algorithm_selection = bool(value)  # type: ignore[attr-defined]

    # -- helpers ---------------------------------------------------------- #
    def reset(self) -> None:
        """Reset all config values to defaults."""
        self._default_backend = "numpy"  # type: ignore[attr-defined]
        self._default_mode = "fast"  # type: ignore[attr-defined]
        self._trace_enabled = False  # type: ignore[attr-defined]
        self._auto_algorithm_selection = True  # type: ignore[attr-defined]

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_backend": self.default_backend,
            "default_mode": self.default_mode,
            "trace_enabled": self.trace_enabled,
            "auto_algorithm_selection": self.auto_algorithm_selection,
        }

    def __repr__(self) -> str:
        return (
            f"GlobalConfig(default_backend={self.default_backend!r}, "
            f"default_mode={self.default_mode!r}, "
            f"trace_enabled={self.trace_enabled!r}, "
            f"auto_algorithm_selection={self.auto_algorithm_selection!r})"
        )


def get_config() -> GlobalConfig:
    """Return the singleton ``GlobalConfig`` instance."""
    return GlobalConfig()
