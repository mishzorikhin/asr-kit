"""In-process tool-call telemetry for the local debug UI."""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar
from typing import Any

current_request_id: ContextVar[str | None] = ContextVar(
    "current_request_id", default=None
)

_counter = itertools.count(1)
_events: deque[dict[str, Any]] = deque(maxlen=500)
_lock = threading.Lock()


def new_request_id() -> str:
    """Allocate a short opaque request id for correlating tool calls."""
    return uuid.uuid4().hex[:12]


def record_tool_call(name: str, status: str = "ok", **details: Any) -> dict[str, Any]:
    """Record an internal tool-call event for the debug UI.

    Args:
        name: Stable event name, for example ``asr.transcribe``.
        status: Outcome status string.
        **details: Optional JSON-serializable event details.

    Returns:
        The recorded event dict.
    """
    event = {
        "id": next(_counter),
        "request_id": current_request_id.get(),
        "time": time.time(),
        "name": name,
        "status": status,
        "details": {key: value for key, value in details.items() if value is not None},
    }
    with _lock:
        _events.append(event)
    return event


def list_tool_calls(limit: int = 200) -> list[dict[str, Any]]:
    """Return the most recent tool-call events.

    Args:
        limit: Maximum number of events to return.

    Returns:
        Newest events up to ``limit``, oldest-first within the window.
    """
    with _lock:
        return list(_events)[-limit:]
