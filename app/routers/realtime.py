from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    REALTIME_WS_IDLE_TIMEOUT_SEC,
)
from app.errors import OpenAIAPIError
from app.model_registry import ModelRegistry
from app.openai_realtime_events import error_event, parse_client_event, session_created_event
from app.services.asr import ASRService
from app.services.realtime_session import RealtimeSession
from app.tool_calls import record_tool_call

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["realtime"])


def _validate_realtime_model(registry: ModelRegistry, model_id: str) -> dict[str, Any]:
    configured = registry.get(model_id)
    if "diarization" in configured["capabilities"] and "transcription" not in configured["capabilities"]:
        raise OpenAIAPIError(
            f"Model '{model_id}' does not support transcription.",
            param="model",
            code="model_not_found",
        )
    return configured


@router.websocket("/realtime")
async def realtime_transcription(
    websocket: WebSocket,
    model: str = Query(..., description="Configured ASR model id from GET /v1/models"),
) -> None:
    await websocket.accept()

    registry: ModelRegistry = websocket.app.state.model_registry
    asr_service: ASRService = websocket.app.state.asr_service

    async def send_event(event: dict[str, Any]) -> None:
        await websocket.send_json(event)

    try:
        configured_model = _validate_realtime_model(registry, model)
    except OpenAIAPIError as exc:
        await send_event(
            error_event(
                exc.message,
                error_type=exc.error_type,
                code=exc.code,
                param=exc.param,
            )
        )
        await websocket.close(code=1008)
        return

    record_tool_call("realtime.connect", model=model)
    session = RealtimeSession(
        asr_service=asr_service,
        model_id=model,
        send_event=send_event,
        language=DEFAULT_LANGUAGE,
        device=DEFAULT_DEVICE,
        compute_type=DEFAULT_COMPUTE_TYPE,
    )
    await send_event(session_created_event(session.session_snapshot()))

    idle_task = asyncio.create_task(_idle_watchdog(websocket, session))

    try:
        while not session.closed:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=REALTIME_WS_IDLE_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                await send_event(
                    error_event(
                        "WebSocket idle timeout",
                        error_type="server_error",
                        code="timeout",
                    )
                )
                break

            session.touch()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError as exc:
                await send_event(
                    error_event(f"Invalid JSON event: {exc}", code="invalid_json")
                )
                continue

            if not isinstance(payload, dict):
                await send_event(error_event("Event payload must be a JSON object"))
                continue

            try:
                event_type, event_payload = parse_client_event(payload)
            except ValueError as exc:
                await send_event(error_event(str(exc)))
                continue

            try:
                await session.handle_event(event_type, event_payload)
            except Exception as exc:
                logger.exception(
                    "Realtime session error model=%s event=%s",
                    configured_model["id"],
                    event_type,
                )
                await send_event(
                    error_event(
                        f"Failed to handle event: {exc}",
                        error_type="server_error",
                        code="session_error",
                    )
                )
    except WebSocketDisconnect:
        logger.info("Realtime client disconnected model=%s", model)
    finally:
        idle_task.cancel()
        await session.close()
        record_tool_call("realtime.disconnect", model=model)


async def _idle_watchdog(websocket: WebSocket, session: RealtimeSession) -> None:
    try:
        while not session.closed:
            await asyncio.sleep(5)
            if session.idle_seconds() >= REALTIME_WS_IDLE_TIMEOUT_SEC:
                await websocket.send_json(
                    error_event(
                        "WebSocket idle timeout",
                        error_type="server_error",
                        code="timeout",
                    )
                )
                await websocket.close(code=1000)
                await session.close()
                return
    except asyncio.CancelledError:
        return
