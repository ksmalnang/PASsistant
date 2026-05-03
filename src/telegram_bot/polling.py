"""Local polling runner for the Telegram bot."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

from src.api.services import chat_service
from src.config.logging import build_logging_config
from src.config.settings import get_settings
from src.telegram_bot.adapter import TelegramBotAdapter

logger = logging.getLogger(__name__)


async def _handle_update(update: Update, context) -> None:
    adapter: TelegramBotAdapter = context.application.bot_data["telegram_adapter"]
    await adapter.handle_update(update)


async def _handle_error(update: object, context) -> None:
    logger.error("Telegram polling handler failed: %s", context.error, exc_info=context.error)


def main() -> None:
    """Run the Telegram bot in long-polling mode."""
    settings = get_settings()
    if not settings.TELEGRAM_ENABLED:
        raise RuntimeError("Telegram support is disabled. Set TELEGRAM_ENABLED=true.")
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be set for polling mode.")

    application = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    application.bot_data["telegram_adapter"] = TelegramBotAdapter(
        chat_service=chat_service,
        bot=application.bot,
        settings=settings,
    )
    application.add_handler(MessageHandler(filters.ALL, _handle_update))
    application.add_error_handler(_handle_error)
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    import logging.config

    logging.config.dictConfig(build_logging_config(get_settings()))
    main()
