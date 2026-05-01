"""Tests for RFC 5424 logging configuration."""

import logging
import re

from src.config.logging import RFC5424Formatter, build_logging_config
from src.config.settings import Settings


def test_rfc5424_formatter_emits_compliant_header():
    """Formatter should emit RFC 5424-style syslog messages."""
    formatter = RFC5424Formatter(
        app_name="student-records-chatbot",
        facility=16,
        environment="test",
    )
    record = logging.LogRecord(
        name="src.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="Policy retrieval succeeded",
        args=(),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert re.match(
        (
            r"^<134>1 "
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z "
            r"[^\s]+ student-records-chatbot \d+ INFO "
            r"\[meta [^\]]*environment=\"test\"[^\]]*\] "
            r"Policy retrieval succeeded$"
        ),
        rendered,
    )


def test_build_logging_config_uses_rfc5424_formatter():
    """dictConfig payload should point handlers at the RFC 5424 formatter."""
    settings = Settings(
        APP_ENV="test",
        LOG_LEVEL="DEBUG",
        LOG_APP_NAME="student-records-chatbot",
        LOG_SYSLOG_FACILITY=16,
    )

    config = build_logging_config(settings)

    assert config["handlers"]["default"]["formatter"] == "rfc5424"
    assert config["root"]["level"] == "DEBUG"
