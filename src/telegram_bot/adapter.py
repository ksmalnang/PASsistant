"""Telegram update adapter for PASsistant."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telegram import Bot, Message, Update
from telegram.constants import ChatAction

from src.config.settings import Settings
from src.telegram_bot.files import (
    TelegramFileDownloadError,
    TelegramFileError,
    TelegramFileTooLargeError,
    TelegramUnsupportedFileTypeError,
    extract_telegram_files,
    get_effective_prompt,
)
from src.telegram_bot.formatting import format_telegram_response, split_telegram_messages

if TYPE_CHECKING:
    from src.api.services import ChatRouteService

logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "Halo! 👋 Butuh bantuan soal layanan akademik atau data mahasiswa? Saya juga bisa membaca PDF atau gambar yang Anda kirim, "
    "dengan atau tanpa caption. 📄"
)
UNSUPPORTED_MESSAGE = "Oops! 😅 Saat ini aku cuma bisa baca teks, PDF, atau gambar ya. 📄"
GENERIC_ERROR_MESSAGE = "Waduh, lagi error nih 😅 Coba kirim ulang sebentar lagi ya!"


class TelegramBotAdapter:
    """Handle Telegram updates using the shared chat service."""

    def __init__(self, chat_service: ChatRouteService, bot: Bot, settings: Settings):
        self._chat_service = chat_service
        self._bot = bot
        self._settings = settings

    async def handle_update(self, update: Update) -> None:
        """Process a Telegram update."""
        message = update.effective_message
        if message is None or message.chat is None:
            self._log_event("Ignoring unsupported Telegram update", update, None, None)
            return

        session_id = self._build_session_id(message.chat.id)
        self._log_event("Received Telegram update", update, session_id, message)

        try:
            if message.text and message.text.strip().startswith(("/start", "/help")):
                await self._send_text(message.chat.id, WELCOME_MESSAGE)
                return

            if message.text and message.text.strip():
                await self._send_typing(message.chat.id)
                response = await self._chat_service.handle_chat_message(
                    message=message.text.strip(),
                    session_id=session_id,
                )
                await self._send_chat_response(
                    message.chat.id,
                    response.response,
                    response.citations,
                )
                self._log_success(
                    update, session_id, message, response.intent, response.documents_processed
                )
                return

            files = await extract_telegram_files(message, self._bot, self._settings)
            if files:
                prompt = get_effective_prompt(message)
                await self._send_typing(message.chat.id)
                response = await self._chat_service.handle_chat_upload(
                    message=prompt,
                    files=files,
                    session_id=session_id,
                )
                await self._send_chat_response(
                    message.chat.id,
                    response.response,
                    response.citations,
                )
                self._log_success(
                    update, session_id, message, response.intent, response.documents_processed
                )
                return

            await self._send_text(message.chat.id, UNSUPPORTED_MESSAGE)
        except TelegramFileTooLargeError as exc:
            await self._send_text(message.chat.id, str(exc))
        except TelegramUnsupportedFileTypeError as exc:
            await self._send_text(message.chat.id, str(exc))
        except TelegramFileDownloadError:
            await self._send_text(message.chat.id, "I could not download that file from Telegram.")
        except TelegramFileError:
            await self._send_text(message.chat.id, UNSUPPORTED_MESSAGE)
        except Exception as exc:
            self._log_event(
                "Telegram adapter failed",
                update,
                session_id,
                message,
                level=logging.ERROR,
                exc_info=exc,
            )
            await self._send_text(message.chat.id, GENERIC_ERROR_MESSAGE)

    def _build_session_id(self, chat_id: int) -> str:
        return f"telegram:{chat_id}"

    async def _send_chat_response(
        self,
        chat_id: int,
        text: str,
        citations: list,
    ) -> None:
        del citations
        formatted = format_telegram_response(text)
        if not formatted:
            formatted = "I processed your request but have no response to provide."
        for chunk in split_telegram_messages(formatted):
            await self._send_text(chat_id, chunk)

    async def _send_typing(self, chat_id: int) -> None:
        await self._bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async def _send_text(self, chat_id: int, text: str) -> None:
        if not text.strip():
            return
        await self._bot.send_message(chat_id=chat_id, text=text)

    def _log_success(
        self,
        update: Update,
        session_id: str,
        message: Message,
        intent: str | None,
        documents_processed: int,
    ) -> None:
        self._log_event(
            "Telegram update handled",
            update,
            session_id,
            message,
            intent=intent,
            documents_processed=documents_processed,
        )

    def _log_event(
        self,
        message: str,
        update: Update,
        session_id: str | None,
        telegram_message: Message | None,
        *,
        level: int = logging.INFO,
        intent: str | None = None,
        documents_processed: int | None = None,
        exc_info: Exception | None = None,
    ) -> None:
        logger.log(
            level,
            message,
            extra={
                "channel": "telegram",
                "telegram_chat_id": getattr(getattr(telegram_message, "chat", None), "id", None),
                "telegram_user_id": getattr(
                    getattr(telegram_message, "from_user", None), "id", None
                ),
                "session_id": session_id,
                "update_id": update.update_id,
                "intent": intent,
                "documents_processed": documents_processed,
            },
            exc_info=exc_info,
        )
