"""Logging setup.

LOG_FORMAT=text (default) -> readable lines for local dev.
LOG_FORMAT=json           -> one JSON object per line for Docker / log
                             aggregators (Datadog, CloudWatch, Loki...).
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Extra fields we attach with logger.info(..., extra={...})
_EXTRA_FIELDS = (
    "event",
    "latency_ms",
    "top_k",
    "num_sources",
    "used_fallback",
    "error_type",
    "query_chars",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the 'metricmedic' logger once. Safe to call repeatedly."""
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.getenv("LOG_FORMAT", "text")).lower()

    logger = logging.getLogger("metricmedic")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:  # already configured (e.g. Streamlit reruns)
        return

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(handler)
