from typing import Any

from fastapi import APIRouter, Request

from app.config import (
    WHISPER_AUTOSCALE_ENABLED,
    WHISPER_MAX_REPLICAS,
    WHISPER_REPLICA_WAIT_SECONDS,
)
from app.services.asr import ASRService

router = APIRouter(prefix="/v1", tags=["status"])


@router.get("/status", summary="Runtime status including Whisper replicas")
def runtime_status(request: Request) -> dict[str, Any]:
    asr_service: ASRService = request.app.state.asr_service
    replicas = asr_service.whisper_replica_status()
    busy = sum(1 for replica in replicas if replica["busy"])
    return {
        "status": "ok",
        "whisper": {
            "autoscale_enabled": WHISPER_AUTOSCALE_ENABLED,
            "max_replicas": WHISPER_MAX_REPLICAS,
            "replica_wait_seconds": WHISPER_REPLICA_WAIT_SECONDS,
            "replicas": replicas,
            "replica_count": len(replicas),
            "busy_replicas": busy,
            "idle_replicas": len(replicas) - busy,
        },
    }
