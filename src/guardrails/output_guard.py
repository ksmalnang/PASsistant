"""Output filtering and PII masking."""

from __future__ import annotations

import re


class OutputGuard:
    """Post-process LLM output to catch prompt leaks and obvious PII."""

    _SYSTEM_LEAK_RESPONSE = (
        "Maaf, saya tidak bisa menjawab permintaan tersebut. "
        "Silakan ajukan pertanyaan tentang layanan akademik."
    )

    SYSTEM_LEAK_PATTERNS = [
        re.compile(r"(?i)(?:system\s+prompt|my\s+instructions?\s+are|I\s+was\s+told\s+to)"),
        re.compile(r"(?i)SECURITY\s*&?\s*SCOPE\s+RULES"),
        re.compile(r"(?i)RESPONSE_SYSTEM_PROMPT|ROUTER_INTENT_PROMPT"),
    ]

    PII_PATTERNS = {
        "nik": re.compile(r"\b\d{16}\b"),
        "nim": re.compile(r"\b\d{3}\.\d{4}\.\d{3,4}\b"),
        "phone": re.compile(r"\b(?:\+62|62|08)\d{8,12}\b"),
        "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    }

    def filter_response(self, text: str) -> str:
        """Filter output for system prompt leaks and redact obvious PII."""
        for pattern in self.SYSTEM_LEAK_PATTERNS:
            if pattern.search(text):
                return self._SYSTEM_LEAK_RESPONSE

        filtered = text
        filtered = self.PII_PATTERNS["nim"].sub("[NIM disamarkan]", filtered)
        filtered = self.PII_PATTERNS["phone"].sub("[nomor telepon disamarkan]", filtered)
        filtered = self.PII_PATTERNS["email"].sub("[email disamarkan]", filtered)
        return filtered
