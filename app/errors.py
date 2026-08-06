"""OpenAI-compatible API error types and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class OpenAIAPIError(Exception):
    """Application error serialized in OpenAI error envelope format."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code


def is_gpu_memory_error(exc: BaseException) -> bool:
    """Return True when an exception looks like GPU memory exhaustion."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "outofmemoryerror",
            "out of memory",
        )
    )


def gpu_memory_error(exc: BaseException) -> OpenAIAPIError:
    """Build a 503 OpenAI-style error for GPU OOM failures."""
    return OpenAIAPIError(
        f"Not enough GPU memory to process this request: {exc}",
        status_code=503,
        error_type="server_error",
        code="insufficient_gpu_memory",
    )


async def openai_error_handler(_: Request, exc: OpenAIAPIError) -> JSONResponse:
    """Serialize ``OpenAIAPIError`` as an OpenAI-compatible JSON body."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Map FastAPI validation errors under ``/v1/`` to OpenAI-style JSON."""
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                }
            },
        )

    return await request_validation_exception_handler(request, exc)
