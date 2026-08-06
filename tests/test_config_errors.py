from __future__ import annotations

import pytest

from app.config import resolve_compute_type, resolve_device
from app.errors import gpu_memory_error, is_gpu_memory_error


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_falls_back_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.torch.cuda.is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("gpu") == "cpu"


def test_resolve_compute_type_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.torch.cuda.is_available", lambda: False)
    assert resolve_compute_type("float16", device="cpu") == "int8"
    assert resolve_compute_type("int8", device="cpu") == "int8"


def test_is_gpu_memory_error_markers() -> None:
    assert is_gpu_memory_error(RuntimeError("CUDA out of memory"))
    assert is_gpu_memory_error(RuntimeError("cuBLAS_STATUS_ALLOC_FAILED"))
    assert not is_gpu_memory_error(RuntimeError("file not found"))


def test_gpu_memory_error_shape() -> None:
    err = gpu_memory_error(RuntimeError("CUDA out of memory"))
    assert err.status_code == 503
    assert err.code == "insufficient_gpu_memory"
    assert "GPU memory" in err.message
