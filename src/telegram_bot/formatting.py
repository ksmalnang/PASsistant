"""Telegram-safe plain-text response formatting."""

from __future__ import annotations

import re

MAX_TELEGRAM_MESSAGE_LENGTH = 4096
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_EMPHASIS_PATTERNS = (
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)"), r"\1"),
    (re.compile(r"(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
)


def format_telegram_response(
    text: str,
) -> str:
    """Normalize Markdown-ish output into mobile-friendly plain text."""
    normalized = _normalize_markdown(text)
    return normalized.strip()


def split_telegram_messages(
    text: str,
    limit: int = MAX_TELEGRAM_MESSAGE_LENGTH,
) -> list[str]:
    """Split a long message into Telegram-sized chunks."""
    content = text.strip()
    if not content:
        return []
    if len(content) <= limit:
        return [content]

    chunks: list[str] = []
    current = ""
    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        chunks.extend(_split_long_block(paragraph, limit))

    if current:
        chunks.append(current)
    return chunks


def _normalize_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    index = 0

    while index < len(lines):
        current = lines[index]
        if _looks_like_table_row(current):
            table_lines: list[str] = []
            while index < len(lines) and _looks_like_table_row(lines[index]):
                table_lines.append(lines[index])
                index += 1
            output.extend(_convert_table_block(table_lines))
            continue

        output.append(_normalize_line(current))
        index += 1

    collapsed = "\n".join(output)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _normalize_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""

    heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
    if heading_match:
        title = _strip_emphasis(heading_match.group(2)).strip(" :")
        return f"{title.upper()}:"

    if stripped.startswith(">"):
        note = stripped.lstrip(">").strip()
        return f"Note: {_strip_emphasis(note)}" if note else ""

    bullet_match = re.match(r"^[-*+]\s+(.*)$", stripped)
    if bullet_match:
        return f"- {_strip_emphasis(bullet_match.group(1))}"

    ordered_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
    if ordered_match:
        return f"{ordered_match.group(1)}. {_strip_emphasis(ordered_match.group(2))}"

    return _strip_emphasis(stripped)


def _strip_emphasis(text: str) -> str:
    normalized = text
    for pattern, replacement in _EMPHASIS_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and stripped.count("|") >= 2


def _convert_table_block(lines: list[str]) -> list[str]:
    rows = [line.strip().strip("|") for line in lines if line.strip()]
    if len(rows) < 2:
        return [_normalize_line(line) for line in lines]

    headers = [_strip_emphasis(cell.strip()) for cell in rows[0].split("|")]
    body_start = 1
    if len(rows) > 1 and _TABLE_SEPARATOR_RE.match(lines[1]):
        body_start = 2

    body_rows = rows[body_start:]
    if not body_rows:
        return [_normalize_line(line) for line in lines]

    converted: list[str] = []
    for row_index, row in enumerate(body_rows, start=1):
        cells = [_strip_emphasis(cell.strip()) for cell in row.split("|")]
        if len(cells) != len(headers):
            return [_normalize_line(line) for line in lines]
        primary = cells[0] if cells and cells[0] else f"Row {row_index}"
        converted.append(f"{row_index}. {primary}")
        for header, value in zip(headers[1:], cells[1:], strict=False):
            if value:
                converted.append(f"{header}: {value}")
        converted.append("")

    while converted and converted[-1] == "":
        converted.pop()
    return converted
def _split_long_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]

    pieces: list[str] = []
    current = ""
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = ""

        if len(line) <= limit:
            current = line
            continue

        words = line.split(" ")
        word_buffer = ""
        for word in words:
            candidate_word = word if not word_buffer else f"{word_buffer} {word}"
            if len(candidate_word) <= limit:
                word_buffer = candidate_word
            else:
                if word_buffer:
                    pieces.append(word_buffer)
                word_buffer = word
        if word_buffer:
            current = word_buffer

    if current:
        pieces.append(current)
    return pieces
