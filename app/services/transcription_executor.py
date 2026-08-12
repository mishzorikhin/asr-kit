from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import queue
import signal
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any

from app.errors import OpenAIAPIError
from app.request_cancel import RequestCancellationToken
from app.tool_calls import current_request_id, record_tool_call

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionJob:
    audio_path: str
    model_id: str
    language: str | None
    prompt: str | None
    temperature: float
    device: str
    compute_type: str
    beam_size: int
    vad_filter: bool
    timestamp_granularities: list[str]
    diarization_enabled: bool
    diarization_model: str
    min_speakers: int | None
    max_speakers: int | None
    num_speakers: int | None
    use_exclusive: bool
    request_id: str | None


class TranscriptionCancelled(Exception):
    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or "cancelled"
        super().__init__(self.reason)


WorkerTarget = Callable[[TranscriptionJob, Any], None]


class SubprocessTranscriptionExecutor:
    def __init__(
        self,
        *,
        poll_interval_seconds: float = 0.25,
        terminate_timeout_seconds: float = 5.0,
        multiprocessing_context: BaseContext | None = None,
        worker_target: WorkerTarget | None = None,
    ) -> None:
        self.poll_interval_seconds = poll_interval_seconds
        self.terminate_timeout_seconds = terminate_timeout_seconds
        self.multiprocessing_context = multiprocessing_context
        self.worker_target = worker_target or _run_transcription_job

    async def run(
        self,
        job: TranscriptionJob,
        cancellation_token: RequestCancellationToken,
    ) -> dict[str, Any]:
        if cancellation_token.cancelled:
            raise TranscriptionCancelled(cancellation_token.reason)

        context = self.multiprocessing_context or multiprocessing.get_context("spawn")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=self.worker_target,
            args=(job, result_queue),
            daemon=False,
        )
        process.start()
        record_tool_call(
            "audio.transcriptions.executor.start",
            mode="subprocess",
            pid=process.pid,
            model=job.model_id,
            diarization=job.diarization_enabled,
        )

        try:
            while True:
                if cancellation_token.cancelled:
                    await self._terminate_process(process, cancellation_token.reason)
                    raise TranscriptionCancelled(cancellation_token.reason)

                try:
                    payload = result_queue.get_nowait()
                except queue.Empty:
                    payload = None

                if payload is not None:
                    process.join(timeout=1)
                    return _decode_worker_payload(payload)

                if not process.is_alive():
                    process.join(timeout=1)
                    try:
                        payload = result_queue.get_nowait()
                    except queue.Empty:
                        payload = None

                    if payload is not None:
                        return _decode_worker_payload(payload)

                    record_tool_call(
                        "audio.transcriptions.executor.exit",
                        status="error",
                        pid=process.pid,
                        exitcode=process.exitcode,
                    )
                    raise OpenAIAPIError(
                        "Transcription worker exited before returning a result.",
                        status_code=500,
                        error_type="server_error",
                        code="transcription_worker_failed",
                    )

                try:
                    await asyncio.wait_for(
                        cancellation_token.wait(),
                        timeout=self.poll_interval_seconds,
                    )
                except TimeoutError:
                    pass
        finally:
            if process.is_alive():
                await self._terminate_process(process, "executor_cleanup")

            process.join(timeout=1)
            result_queue.close()
            result_queue.join_thread()

    async def _terminate_process(
        self,
        process: BaseProcess,
        reason: str | None,
    ) -> None:
        if not process.is_alive():
            return

        pid = process.pid
        record_tool_call(
            "audio.transcriptions.executor.terminate",
            status="cancelled",
            pid=pid,
            reason=reason,
        )
        logger.info("Terminating transcription worker pid=%s reason=%s", pid, reason)
        _signal_process_group_or_process(process, signal.SIGTERM)

        deadline = asyncio.get_running_loop().time() + self.terminate_timeout_seconds
        while process.is_alive() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

        if process.is_alive():
            record_tool_call(
                "audio.transcriptions.executor.kill",
                status="cancelled",
                pid=pid,
                reason=reason,
            )
            logger.warning("Killing transcription worker pid=%s reason=%s", pid, reason)
            _signal_process_group_or_process(process, signal.SIGKILL)

        process.join(timeout=1)


def _signal_process_group_or_process(process: BaseProcess, sig: signal.Signals) -> None:
    pid = process.pid
    if pid is None:
        return

    try:
        os.killpg(pid, sig)
        return
    except ProcessLookupError:
        # The worker may not have created its process group yet; signal the
        # process directly in that startup race.
        pass
    except PermissionError:
        logger.warning("Cannot signal transcription process group pid=%s signal=%s", pid, sig)
    except OSError:
        # The worker may not have entered its own process group yet.
        pass

    if sig == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _run_transcription_job(job: TranscriptionJob, result_queue: Any) -> None:
    from app.model_registry import ModelRegistry
    from app.services.asr import ASRService
    from app.services.diarization import DiarizationService

    _start_own_process_group()
    request_token = current_request_id.set(job.request_id)

    try:
        registry = ModelRegistry()
        asr_service = ASRService(registry)

        if job.diarization_enabled:
            diarization_service = DiarizationService(asr_service)
            transcription = diarization_service.transcribe_with_diarization(
                job.audio_path,
                model_id=job.model_id,
                language=job.language,
                prompt=job.prompt,
                temperature=job.temperature,
                device=job.device,
                compute_type=job.compute_type,
                beam_size=job.beam_size,
                vad_filter=job.vad_filter,
                timestamp_granularities=job.timestamp_granularities,
                diarization_model=job.diarization_model,
                min_speakers=job.min_speakers,
                max_speakers=job.max_speakers,
                num_speakers=job.num_speakers,
                use_exclusive=job.use_exclusive,
            )
        else:
            transcription = asr_service.transcribe(
                job.audio_path,
                model_id=job.model_id,
                language=job.language,
                prompt=job.prompt,
                temperature=job.temperature,
                device=job.device,
                compute_type=job.compute_type,
                beam_size=job.beam_size,
                vad_filter=job.vad_filter,
                timestamp_granularities=job.timestamp_granularities,
            )

        result_queue.put({"status": "ok", "result": transcription})
    except OpenAIAPIError as exc:
        result_queue.put(
            {
                "status": "openai_error",
                "message": exc.message,
                "status_code": exc.status_code,
                "error_type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "status": "error",
                "message": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        current_request_id.reset(request_token)


def _start_own_process_group() -> None:
    if not hasattr(os, "setsid"):
        return

    try:
        os.setsid()
    except OSError:
        logger.debug("Could not create transcription worker process group", exc_info=True)


def _decode_worker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status == "ok":
        record_tool_call("audio.transcriptions.executor.complete", mode="subprocess")
        return payload["result"]

    if status == "openai_error":
        raise OpenAIAPIError(
            payload["message"],
            status_code=payload["status_code"],
            error_type=payload["error_type"],
            param=payload.get("param"),
            code=payload.get("code"),
        )

    record_tool_call(
        "audio.transcriptions.executor.error",
        status="error",
        error=payload.get("message"),
        exception_type=payload.get("exception_type"),
    )
    logger.error(
        "Transcription worker failed: %s\n%s",
        payload.get("message"),
        payload.get("traceback", ""),
    )
    raise OpenAIAPIError(
        f"Transcription worker failed: {payload.get('message')}",
        status_code=500,
        error_type="server_error",
        code="transcription_worker_error",
    )
