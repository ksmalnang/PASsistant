"""WebSocket endpoints."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.sessions import get_or_create_agent
from src.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)
router = APIRouter()
_guard = InputGuard()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    agent, _ = get_or_create_agent(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            if not websocket.app.state.rate_limiter.allow(
                getattr(getattr(websocket, "client", None), "host", None) or "unknown"
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": "Rate limit exceeded. Please try again later.",
                    }
                )
                continue
            guard_result = _guard.validate(data)
            if not guard_result.safe:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": f"Message rejected: {guard_result.reason}",
                    }
                )
                continue
            async for update in agent.stream_chat(guard_result.sanitized or data.strip()):
                await websocket.send_json(
                    {
                        "type": "stream_update",
                        "data": str(update),
                    }
                )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        await websocket.close(code=1011)
