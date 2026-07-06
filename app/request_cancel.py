from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request

from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)


class RequestCancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str) -> None:
        if self._event.is_set():
            return

        self._reason = reason
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


async def watch_request_disconnect(
    request: "Request" | Any,
    cancellation_token: RequestCancellationToken,
    *,
    poll_interval_seconds: float = 0.25,
) -> None:
    while not cancellation_token.cancelled:
        if await request.is_disconnected():
            cancellation_token.cancel("client_disconnected")
            record_tool_call(
                "http.client_disconnect",
                status="cancelled",
                path=request.url.path,
            )
            logger.info("Client disconnected path=%s", request.url.path)
            return

        await asyncio.sleep(poll_interval_seconds)
