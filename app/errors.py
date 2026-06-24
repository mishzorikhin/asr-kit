from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class OpenAIAPIError(Exception):
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


def is_cuda_unavailable_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "no cuda-capable device is detected",
            "cuda driver version is insufficient",
            "cuda error: no device",
            "found no nvidia driver",
        )
    )


def gpu_memory_error(exc: BaseException) -> OpenAIAPIError:
    return OpenAIAPIError(
        f"Not enough GPU memory to process this request: {exc}",
        status_code=503,
        error_type="server_error",
        code="insufficient_gpu_memory",
    )


async def openai_error_handler(_: Request, exc: OpenAIAPIError) -> JSONResponse:
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
