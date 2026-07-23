"""Structured logging setup shared across the extraction-service process.

Writes two sinks:
- stdout, human-readable, for ``docker compose logs``;
- a rotating JSON-lines file on a persistent volume, so history survives
  container redeploys (unlike stdout, which docker's json-file driver
  discards when the container is recreated).

Call :func:`configure_logging` once at process startup, before any other
module obtains a logger. Steps within a single analysis should log through
:func:`bind_context`/``extra={"analysis_id": ..., "file_id": ..., "step": ...}``
so log lines can be correlated back to one document run.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from typing import Any

_CONTEXT_FIELDS = ("analysis_id", "file_id", "file_type", "step")


class JsonFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(service_name: str) -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            " | analysis_id=%(analysis_id)s file_id=%(file_id)s step=%(step)s"
        )
    )
    console_handler.addFilter(_DefaultContextFilter())
    root.addHandler(console_handler)

    log_dir = os.getenv("LOG_DIR", "/var/log/app")
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, f"{service_name}.log"),
            maxBytes=50 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not create log directory %s; file logging disabled", log_dir
        )


class _DefaultContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in _CONTEXT_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, "-")
        return True
