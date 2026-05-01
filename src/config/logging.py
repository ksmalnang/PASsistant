"""Centralized RFC 5424 logging configuration."""

from __future__ import annotations

import logging
import logging.config
import os
import socket
from datetime import UTC, datetime
from itertools import count
from typing import Any

from src.config.settings import Settings, get_settings

_SEQUENCE = count(1)


class RFC5424Formatter(logging.Formatter):
    """Render log records as RFC 5424 syslog messages."""

    _SEVERITY_MAP = {
        logging.CRITICAL: 2,
        logging.ERROR: 3,
        logging.WARNING: 4,
        logging.INFO: 6,
        logging.DEBUG: 7,
    }

    def __init__(self, app_name: str, facility: int, environment: str):
        super().__init__()
        self.app_name = self._clean_header_value(app_name, max_length=48)
        self.facility = facility
        self.environment = environment
        self.hostname = self._clean_header_value(socket.gethostname(), max_length=255)
        self.procid = str(os.getpid())

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record according to RFC 5424."""
        severity = self._get_severity(record.levelno)
        priority = (self.facility * 8) + severity
        timestamp = self._format_timestamp(record.created)
        msgid = self._resolve_msgid(record)
        structured_data = self._build_structured_data(record)
        message = self._sanitize_message(record.getMessage())

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            if exception_text:
                message = f"{message}\n{exception_text}" if message else exception_text

        return (
            f"<{priority}>1 {timestamp} {self.hostname} {self.app_name} "
            f"{self.procid} {msgid} {structured_data} {message}"
        )

    def _get_severity(self, levelno: int) -> int:
        """Map Python levels to syslog severities."""
        if levelno >= logging.CRITICAL:
            return self._SEVERITY_MAP[logging.CRITICAL]
        if levelno >= logging.ERROR:
            return self._SEVERITY_MAP[logging.ERROR]
        if levelno >= logging.WARNING:
            return self._SEVERITY_MAP[logging.WARNING]
        if levelno >= logging.INFO:
            return self._SEVERITY_MAP[logging.INFO]
        return self._SEVERITY_MAP[logging.DEBUG]

    def _format_timestamp(self, created: float) -> str:
        """Render a RFC 3339 timestamp with UTC offset."""
        return (
            datetime.fromtimestamp(created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _resolve_msgid(self, record: logging.LogRecord) -> str:
        """Resolve a safe RFC 5424 MSGID value."""
        raw_msgid = getattr(record, "msgid", record.levelname)
        return self._clean_header_value(str(raw_msgid), max_length=32)

    def _build_structured_data(self, record: logging.LogRecord) -> str:
        """Build the structured-data element for the record."""
        sequence_id = getattr(record, "sequence_id", next(_SEQUENCE))
        params = {
            "sequenceId": str(sequence_id),
            "logger": record.name,
            "module": record.module,
            "line": str(record.lineno),
            "environment": self.environment,
        }
        rendered = " ".join(
            f'{key}="{self._escape_sd_param(value)}"'
            for key, value in params.items()
            if value
        )
        return f"[meta {rendered}]"

    def _sanitize_message(self, message: str) -> str:
        """Ensure the message is printable for the syslog payload."""
        if not message:
            return "-"
        return "".join(
            character if character.isprintable() or character in "\n\t" else " "
            for character in message
        )

    def _escape_sd_param(self, value: str) -> str:
        """Escape structured-data parameter values."""
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("]", "\\]")
        )

    def _clean_header_value(self, value: str, max_length: int) -> str:
        """Normalize header fields to RFC 5424-safe ASCII tokens."""
        cleaned = "".join(
            character if 33 <= ord(character) <= 126 and character != " " else "_"
            for character in value
        ).strip("_")
        if not cleaned:
            return "-"
        return cleaned[:max_length]


def build_logging_config(settings: Settings | None = None) -> dict[str, Any]:
    """Build a dictConfig payload for application logging."""
    resolved_settings = settings or get_settings()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "rfc5424": {
                "()": "src.config.logging.RFC5424Formatter",
                "app_name": resolved_settings.LOG_APP_NAME,
                "facility": resolved_settings.LOG_SYSLOG_FACILITY,
                "environment": resolved_settings.APP_ENV,
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "rfc5424",
                "stream": "ext://sys.__stdout__",
            }
        },
        "root": {
            "level": resolved_settings.LOG_LEVEL.upper(),
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {"level": resolved_settings.LOG_LEVEL.upper(), "propagate": True},
            "uvicorn.error": {
                "level": resolved_settings.LOG_LEVEL.upper(),
                "propagate": True,
            },
            "uvicorn.access": {
                "level": resolved_settings.LOG_LEVEL.upper(),
                "propagate": True,
            },
        },
    }


def configure_logging(settings: Settings | None = None) -> None:
    """Apply centralized logging configuration."""
    logging.config.dictConfig(build_logging_config(settings))
