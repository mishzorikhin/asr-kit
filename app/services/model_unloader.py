"""Background idle unload of ASR and diarization models."""

from __future__ import annotations

import gc
import logging
import threading

from app.config import MODEL_IDLE_TTL_SECONDS, MODEL_UNLOAD_INTERVAL_SECONDS
from app.services.asr import ASRService
from app.services.diarization import DiarizationService

logger = logging.getLogger(__name__)


class ModelUnloader:
    """Periodically unloads idle models to reclaim GPU/CPU memory."""

    def __init__(
        self,
        asr_service: ASRService,
        diarization_service: DiarizationService,
    ) -> None:
        self.asr_service = asr_service
        self.diarization_service = diarization_service
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background unload loop when TTL is enabled."""
        if MODEL_IDLE_TTL_SECONDS <= 0:
            logger.info("Idle model unloading is disabled")
            return

        if self._thread is not None:
            return

        logger.info(
            "Starting idle model unloader ttl=%ss interval=%ss",
            MODEL_IDLE_TTL_SECONDS,
            self._interval_seconds,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="model-unloader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background unload loop."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self._interval_seconds)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            unloaded = self.unload_idle()
            if unloaded:
                gc.collect()

    def unload_idle(self) -> int:
        """Unload idle ASR and diarization models once.

        Returns:
            Total number of models/pipelines unloaded.
        """
        asr_unloaded = self.asr_service.unload_idle_models()
        diarization_unloaded = self.diarization_service.unload_idle_pipelines()
        return asr_unloaded + diarization_unloaded

    @property
    def _interval_seconds(self) -> int:
        return max(1, MODEL_UNLOAD_INTERVAL_SECONDS)
