import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.config import SUPPORTED_AUDIO_EXTENSIONS
from app.errors import OpenAIAPIError


async def save_upload_to_temp_file(file: UploadFile) -> str:
    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix and suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise OpenAIAPIError(
            (
                "Unsupported audio file extension. Supported formats: "
                + ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            ),
            param="file",
            code="unsupported_file",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".audio") as tmp:
        tmp_path = tmp.name
        await file.seek(0)
        shutil.copyfileobj(file.file, tmp)

    file_size = Path(tmp_path).stat().st_size
    if file_size == 0:
        Path(tmp_path).unlink(missing_ok=True)
        raise OpenAIAPIError(
            "Uploaded audio file is empty",
            param="file",
            code="empty_file",
        )
    return tmp_path
