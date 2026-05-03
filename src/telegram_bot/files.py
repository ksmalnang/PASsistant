"""Telegram file download and validation helpers."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from telegram import Bot, Document, Message, PhotoSize
from telegram.error import TelegramError

from src.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}


class TelegramFileError(Exception):
    """Base exception for Telegram file handling failures."""


class TelegramFileTooLargeError(TelegramFileError):
    """Raised when a Telegram file exceeds the configured size limit."""


class TelegramUnsupportedFileTypeError(TelegramFileError):
    """Raised when a Telegram file type is unsupported."""


class TelegramFileDownloadError(TelegramFileError):
    """Raised when Telegram file download fails."""


async def extract_telegram_files(
    message: Message,
    bot: Bot,
    settings: Settings,
) -> list[tuple[str, bytes]]:
    """Download and validate files from a Telegram message."""
    document = message.document
    if document is not None:
        return [await _download_document(document, bot, settings)]

    photos = message.photo or []
    if photos:
        return [await _download_photo(photos[-1], bot, settings)]

    return []


def get_effective_prompt(message: Message, default: str = "Process this document.") -> str:
    """Resolve caption text or a default prompt for document/image uploads."""
    caption = (message.caption or "").strip()
    return caption or default


async def _download_document(
    document: Document,
    bot: Bot,
    settings: Settings,
) -> tuple[str, bytes]:
    mime_type = document.mime_type or _guess_mime_type(document.file_name)
    _validate_file_metadata(
        file_size=document.file_size,
        mime_type=mime_type,
        settings=settings,
    )
    filename = document.file_name or f"document{_suffix_for_mime_type(mime_type)}"
    return await _download_file(
        file_reference=document,
        filename=filename,
        mime_type=mime_type,
        bot=bot,
        settings=settings,
    )


async def _download_photo(
    photo: PhotoSize,
    bot: Bot,
    settings: Settings,
) -> tuple[str, bytes]:
    mime_type = "image/jpeg"
    _validate_file_metadata(
        file_size=photo.file_size,
        mime_type=mime_type,
        settings=settings,
    )
    filename = f"photo_{photo.file_unique_id or photo.file_id}.jpg"
    return await _download_file(
        file_reference=photo,
        filename=filename,
        mime_type=mime_type,
        bot=bot,
        settings=settings,
    )


async def _download_file(
    file_reference: Document | PhotoSize,
    filename: str,
    mime_type: str | None,
    bot: Bot,
    settings: Settings,
) -> tuple[str, bytes]:
    try:
        telegram_file = await bot.get_file(file_reference.file_id)
        data = await telegram_file.download_as_bytearray()
    except TelegramError as exc:
        raise TelegramFileDownloadError("Failed to download the file from Telegram.") from exc

    _validate_file_metadata(
        file_size=len(data),
        mime_type=mime_type,
        settings=settings,
    )
    return filename, bytes(data)


def _validate_file_metadata(
    file_size: int | None,
    mime_type: str | None,
    settings: Settings,
) -> None:
    if file_size is not None and file_size > settings.TELEGRAM_MAX_FILE_BYTES:
        raise TelegramFileTooLargeError(
            f"File is too large. Maximum supported size is {settings.TELEGRAM_MAX_FILE_BYTES} bytes."
        )

    allowed_mime_types = set(
        settings.TELEGRAM_ALLOWED_FILE_MIME_TYPES or DEFAULT_ALLOWED_MIME_TYPES
    )
    if mime_type and mime_type not in allowed_mime_types:
        raise TelegramUnsupportedFileTypeError(
            "Unsupported file type. Please send a PDF, JPEG, PNG, or WEBP image."
        )


def _guess_mime_type(filename: str | None) -> str | None:
    if not filename:
        return None
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type


def _suffix_for_mime_type(mime_type: str | None) -> str:
    guessed = mimetypes.guess_extension(mime_type or "")
    if guessed:
        return guessed
    return Path("upload").suffix
