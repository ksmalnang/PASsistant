"""Telegram webhook endpoint."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException, Request, status
from telegram import Bot, Update

from src.api.models import ErrorResponse
from src.api.services import chat_service
from src.config.settings import Settings, get_settings
from src.telegram_bot.adapter import TelegramBotAdapter

router = APIRouter()


@lru_cache
def get_telegram_bot() -> Bot:
    """Build a reusable Telegram bot client."""
    settings = get_settings()
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be set when Telegram support is enabled.")
    return Bot(token=settings.TELEGRAM_BOT_TOKEN)


@lru_cache
def get_telegram_adapter() -> TelegramBotAdapter:
    """Build a reusable Telegram adapter."""
    return TelegramBotAdapter(
        chat_service=chat_service,
        bot=get_telegram_bot(),
        settings=get_settings(),
    )


def _verify_telegram_request(
    settings: Settings,
    secret_token: str | None,
) -> None:
    if not settings.TELEGRAM_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Telegram is disabled.")

    if settings.is_production and not settings.TELEGRAM_WEBHOOK_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Telegram webhook secret is not configured.",
        )

    expected_secret = settings.TELEGRAM_WEBHOOK_SECRET_TOKEN
    if settings.is_production and expected_secret and secret_token != expected_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Telegram webhook secret.",
        )


@router.post(
    "/telegram/webhook",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "The Telegram update payload is invalid.",
        },
        status.HTTP_403_FORBIDDEN: {
            "model": ErrorResponse,
            "description": "The Telegram webhook secret is invalid or missing.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Telegram support is disabled.",
        },
    },
)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, bool]:
    """Handle Telegram webhook updates."""
    settings = get_settings()
    _verify_telegram_request(settings, x_telegram_bot_api_secret_token)

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed Telegram update payload.",
        ) from exc

    try:
        update = Update.de_json(payload, get_telegram_bot())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed Telegram update payload.",
        ) from exc

    await get_telegram_adapter().handle_update(update)
    return {"ok": True}
