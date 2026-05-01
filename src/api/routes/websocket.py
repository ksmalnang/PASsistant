"""WebSocket endpoints."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.sessions import get_or_create_agent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    agent, _ = get_or_create_agent(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            async for update in agent.stream_chat(data):
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
