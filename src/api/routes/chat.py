"""Chat endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.api.models import ChatRequest, ChatResponse, ErrorResponse
from src.api.services import handle_chat_message, handle_chat_upload

logger = logging.getLogger(__name__)
router = APIRouter()
UPLOAD_FILE_PARAM = File(...)
CHAT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The request payload is invalid.",
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
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a chat message to the agent."""
    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message must not be empty.",
            )
        return await handle_chat_message(
            message=request.message,
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
    message: str = "Process this document",
    files: list[UploadFile] = UPLOAD_FILE_PARAM,
    session_id: str | None = None,
) -> ChatResponse:
    """Chat with document uploads."""
    try:
        if not message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message must not be empty.",
            )
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file must be uploaded.",
            )
        return await handle_chat_upload(
            message=message,
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
