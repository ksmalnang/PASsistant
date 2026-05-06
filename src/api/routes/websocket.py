"""WebSocket endpoints."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.models import ChatStreamEvent
from src.api.services import chat_service
from src.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)
router = APIRouter()
_guard = InputGuard()


@router.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    try:
        while True:
            payload = await _receive_chat_payload(websocket)
            message = payload["message"]
            last_event_id = payload.get("last_event_id")

            if not websocket.app.state.rate_limiter.allow(
                getattr(getattr(websocket, "client", None), "host", None) or "unknown"
            ):
                await websocket.send_json(
                    _build_socket_error_event(
                        thread_id=thread_id,
                        message="Rate limit exceeded. Please try again later.",
                    ).model_dump(mode="json")
                )
                continue

            guard_result = _guard.validate(message)
            if not guard_result.safe:
                await websocket.send_json(
                    _build_socket_error_event(
                        thread_id=thread_id,
                        message=f"Message rejected: {guard_result.reason}",
                    ).model_dump(mode="json")
                )
                continue

            async for event in chat_service.stream_chat_message(
                message=guard_result.sanitized or message.strip(),
                thread_id=thread_id,
                last_event_id=last_event_id,
            ):
                await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", thread_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        await websocket.close(code=1011)


async def _receive_chat_payload(websocket: WebSocket) -> dict[str, Any]:
    """Accept either JSON messages or raw text messages."""
    incoming = await websocket.receive()

    text = incoming.get("text")
    if text is None:
        return {"message": ""}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}

    if isinstance(payload, str):
        return {"message": payload}
    if isinstance(payload, dict):
        return {
            "message": str(payload.get("message", "")),
            "last_event_id": payload.get("last_event_id"),
        }
    return {"message": text}


def _build_socket_error_event(thread_id: str, message: str) -> ChatStreamEvent:
    """Build a typed websocket error event."""
    return ChatStreamEvent(
        event_id=f"rejected:{uuid.uuid4()}",
        event_type="run.failed",
        thread_id=thread_id,
        run_id="rejected",
        timestamp=datetime.now(UTC),
        sequence=0,
        data={
            "code": "REQUEST_REJECTED",
            "message": message,
            "retryable": False,
        },
    )
