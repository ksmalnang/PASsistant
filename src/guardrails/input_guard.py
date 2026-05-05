"""Input sanitization and prompt injection detection."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class InputGuardResult:
    """Result of input validation."""

    def __init__(
        self,
        safe: bool,
        reason: str | None = None,
        sanitized: str | None = None,
    ) -> None:
        self.safe = safe
        self.reason = reason
        self.sanitized = sanitized


class InputGuard:
    """Detect and block prompt injection, jailbreaks, and abuse patterns."""

    MAX_MESSAGE_LENGTH = 4000
    MIN_MESSAGE_LENGTH = 1

    INJECTION_PATTERNS = [
        r"(?i)\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|rules?|context)",
        r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\b",
        r"(?i)\b(?:new\s+instruction|override\s+instruction|system\s*:\s*)",
        r"(?i)\b(?:show|print|output|reveal|repeat|display)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instruction|rules?|configuration)",
        r"(?i)\b(?:show|print|output|reveal|repeat|display)\s+your\s+(?:prompt|instructions?|rules?|configuration)\b",
        r"(?i)\bwhat\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instruction|rules?)\b",
        r"(?i)\bwhat\s+(?:are|is)\s+your\s+instructions?\b",
        r"(?i)```\s*system",
        r"(?i)<\|(?:im_start|system|endoftext)\|>",
        r"(?i)\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>",
        r"(?i)\bDAN\b.*(?:mode|jailbreak|unlocked)",
        r"(?i)\bdo\s+anything\s+now\b",
    ]

    SCOPE_VIOLATION_PATTERNS = [
        r"(?i)\b(?:write\s+(?:me\s+)?(?:a\s+)?(?:code|script|program|essay|story|poem))\b",
        r"(?i)\b(?:hack|exploit|bypass\s+security|sql\s+injection)\b",
    ]

    _compiled_injection = [re.compile(pattern) for pattern in INJECTION_PATTERNS]
    _compiled_scope = [re.compile(pattern) for pattern in SCOPE_VIOLATION_PATTERNS]

    def validate(self, message: str) -> InputGuardResult:
        """Validate user input and return a safety assessment."""
        stripped = message.strip()
        if len(stripped) < self.MIN_MESSAGE_LENGTH:
            return InputGuardResult(safe=False, reason="empty_message")

        if len(message) > self.MAX_MESSAGE_LENGTH:
            return InputGuardResult(safe=False, reason="message_too_long")

        for pattern in self._compiled_injection:
            if pattern.search(message):
                logger.warning(
                    "Prompt injection detected",
                    extra={"pattern": pattern.pattern, "input_preview": message[:100]},
                )
                return InputGuardResult(safe=False, reason="prompt_injection")

        for pattern in self._compiled_scope:
            if pattern.search(message):
                logger.info("Scope violation detected", extra={"input_preview": message[:100]})
                break

        return InputGuardResult(safe=True, sanitized=stripped)
