from __future__ import annotations

import asyncio
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _import_or_skip(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise unittest.SkipTest(f"Cannot import {module_name}: {exc}") from exc


def _sleeping_worker(_job, _result_queue) -> None:
    pid_file = os.environ.get("TRANSCRIPTION_TEST_PID_FILE")
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()), encoding="utf-8")

    while True:
        time.sleep(0.1)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class SubprocessTranscriptionExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_terminates_worker_process(self) -> None:
        transcription_executor = _import_or_skip("app.services.transcription_executor")
        request_cancel = _import_or_skip("app.request_cancel")

        try:
            multiprocessing_context = multiprocessing.get_context("fork")
        except ValueError as exc:
            raise unittest.SkipTest("fork multiprocessing context is unavailable") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "worker.pid"
            os.environ["TRANSCRIPTION_TEST_PID_FILE"] = str(pid_file)
            try:
                token = request_cancel.RequestCancellationToken()
                executor = transcription_executor.SubprocessTranscriptionExecutor(
                    poll_interval_seconds=0.05,
                    terminate_timeout_seconds=0.2,
                    multiprocessing_context=multiprocessing_context,
                    worker_target=_sleeping_worker,
                )
                job = transcription_executor.TranscriptionJob(
                    audio_path="/tmp/audio.wav",
                    model_id="test-model",
                    language=None,
                    prompt=None,
                    temperature=0.0,
                    device="cpu",
                    compute_type="int8",
                    beam_size=1,
                    vad_filter=False,
                    timestamp_granularities=["segment"],
                    diarization_enabled=False,
                    diarization_model="/tmp/diarization",
                    min_speakers=None,
                    max_speakers=None,
                    num_speakers=None,
                    use_exclusive=True,
                    request_id="test-request",
                )

                task = asyncio.create_task(executor.run(job, token))
                for _ in range(50):
                    if pid_file.exists():
                        break
                    await asyncio.sleep(0.05)

                self.assertTrue(pid_file.exists())
                pid = int(pid_file.read_text(encoding="utf-8"))

                token.cancel("client_disconnected")
                with self.assertRaises(transcription_executor.TranscriptionCancelled):
                    await asyncio.wait_for(task, timeout=5)

                self.assertFalse(_pid_exists(pid))
            finally:
                os.environ.pop("TRANSCRIPTION_TEST_PID_FILE", None)


class AudioRouterCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_transcription_cleans_temp_upload(self) -> None:
        audio = _import_or_skip("app.routers.audio")
        transcription_executor = _import_or_skip("app.services.transcription_executor")

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = Path(tmp_file.name)
        tmp_file.write(b"audio")
        tmp_file.close()

        class FakeForm:
            def getlist(self, _name: str) -> list[str]:
                return []

        class FakeRegistry:
            def get(self, model_id: str) -> dict[str, object]:
                return {
                    "id": model_id,
                    "path": "/tmp/model",
                    "capabilities": {"transcription"},
                    "diarization_model": "/tmp/diarization",
                }

        class CancellingExecutor:
            async def run(self, job, _cancellation_token):
                self.job = job
                self.saw_file_before_cancel = Path(job.audio_path).exists()
                raise transcription_executor.TranscriptionCancelled("client_disconnected")

        class FakeRequest:
            def __init__(self) -> None:
                self.url = SimpleNamespace(path="/v1/audio/transcriptions")
                self.app = SimpleNamespace(
                    state=SimpleNamespace(
                        model_registry=FakeRegistry(),
                        transcription_executor=CancellingExecutor(),
                    )
                )

            async def form(self) -> FakeForm:
                return FakeForm()

            async def is_disconnected(self) -> bool:
                return False

        request = FakeRequest()
        upload = SimpleNamespace(filename="sample.wav")

        try:
            with patch.object(audio, "save_upload_to_temp_file", return_value=str(tmp_path)):
                response = await audio.create_transcription(
                    request,
                    file=upload,
                    model="test-model",
                    language=None,
                    prompt=None,
                    response_format="json",
                    temperature=0.0,
                    stream=False,
                    include=None,
                    timestamp_granularities=None,
                    known_speaker_names=None,
                    known_speaker_references=None,
                    chunking_strategy=None,
                    device="cpu",
                    compute_type="int8",
                    beam_size=1,
                    vad_filter=False,
                    diarization_model="/tmp/diarization",
                    min_speakers=None,
                    max_speakers=None,
                    num_speakers=None,
                    use_exclusive=True,
                )
        finally:
            tmp_path.unlink(missing_ok=True)

        executor = request.app.state.transcription_executor
        self.assertEqual(response.status_code, 499)
        self.assertTrue(executor.saw_file_before_cancel)
        self.assertFalse(tmp_path.exists())
