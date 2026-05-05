"""Chat endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from src.api.models import ChatRequest, ChatResponse, ErrorResponse
from src.api.services import handle_chat_message, handle_chat_upload
from src.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)
router = APIRouter()
_guard = InputGuard()
UPLOAD_FILE_PARAM = File(...)
CHAT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The request payload is invalid.",
    },
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "model": ErrorResponse,
        "description": "The client exceeded the request rate limit.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "The server failed to process the chat request.",
    },
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Submit a text message to the chatbot and receive a structured reply.",
    responses=CHAT_ERROR_RESPONSES,
)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """Send a chat message to the agent."""
    try:
        _enforce_rate_limit(http_request)
        guard_result = _guard.validate(request.message)
        if not guard_result.safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Message rejected: {guard_result.reason}",
            )
        return await handle_chat_message(
            message=guard_result.sanitized or request.message.strip(),
            session_id=request.session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request.",
        ) from exc


@router.post(
    "/chat/upload",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message with file uploads",
    description=(
        "Upload one or more documents together with a prompt so the chatbot can "
        "process the files in the same conversation request. This endpoint is "
        "for chat-context file processing and does not ingest documents into the "
        "shared retrieval index."
    ),
    responses=CHAT_ERROR_RESPONSES,
)
async def chat_with_upload(
    http_request: Request,
    message: str = "Process this document",
    files: list[UploadFile] = UPLOAD_FILE_PARAM,
    session_id: str | None = None,
) -> ChatResponse:
    """Chat with document uploads."""
    try:
        _enforce_rate_limit(http_request)
        guard_result = _guard.validate(message)
        if not guard_result.safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Message rejected: {guard_result.reason}",
            )
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file must be uploaded.",
            )
        return await handle_chat_upload(
            message=guard_result.sanitized or message.strip(),
            files=files,
            session_id=session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload chat endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded chat documents.",
        ) from exc


def _enforce_rate_limit(request: Request) -> None:
    """Reject requests that exceed the in-memory rate limit."""
    client_host = getattr(request.client, "host", None) or "unknown"
    limiter = request.app.state.rate_limiter
    if not limiter.allow(client_host):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )
