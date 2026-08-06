"""Logging bootstrap for the ASR server process."""

from __future__ import annotations

import logging

from app.config import LOG_LEVEL


def configure_logging() -> None:
    """Configure root logging once at application startup."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
