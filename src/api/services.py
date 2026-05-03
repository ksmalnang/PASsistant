"""Service helpers shared across route modules."""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import UploadFile
from langchain_core.messages import AIMessage

from src.api.helpers import create_document_processor, read_upload_files
from src.api.models import ChatResponse, DocumentIngestionResponse
from src.api.sessions import get_or_create_agent
from src.services.contracts import ChatAgent, DocumentProcessor, SessionManager

logger = logging.getLogger(__name__)


class ChatRouteService:
    """Coordinate chat API requests with session-backed agents."""

    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    async def handle_chat_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Send a plain chat message to an agent session."""
        agent, active_session_id = self._session_manager.get_or_create(session_id)
        final_state = await agent.chat_with_state(message)
        return ChatResponse(
            response=self._extract_response_text(final_state),
            session_id=active_session_id,
            intent=final_state.current_intent,
            citations=self._extract_citations(final_state),
        )

    async def handle_chat_upload(
        self,
        message: str,
        files: list[UploadFile],
        session_id: str | None = None,
    ) -> ChatResponse:
        """Send uploaded files to an agent session."""
        agent, active_session_id = self._session_manager.get_or_create(session_id)
        file_tuples = await read_upload_files(files)
        final_state = await agent.chat_with_state(message, files=file_tuples)
        return ChatResponse(
            response=self._extract_response_text(final_state),
            session_id=active_session_id,
            intent=final_state.current_intent,
            documents_processed=len(files),
            citations=self._extract_citations(final_state),
        )

    def _extract_response_text(self, final_state: Any) -> str:
        """Resolve the assistant response text from a workflow state."""
        if getattr(final_state, "draft_response", None):
            return str(final_state.draft_response)

        messages = list(getattr(final_state, "messages", []) or [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content)

        if messages:
            return str(messages[-1].content)
        return "I processed your request but have no response to provide."

    def _extract_citations(self, final_state: Any) -> list[Any]:
        """Resolve structured citations from a workflow state or test double."""
        return list(getattr(final_state, "citations", []) or [])


class DocumentRouteService:
    """Coordinate document-upload API requests."""

    def __init__(
        self,
        processor_factory: Callable[[], DocumentProcessor],
    ):
        self._processor_factory = processor_factory

    async def handle_knowledge_base_ingestion(
        self,
        files: list[UploadFile],
    ) -> list[DocumentIngestionResponse]:
        """Ingest uploaded documents into the retrieval stack."""
        processor = self._processor_factory()
        responses: list[DocumentIngestionResponse] = []

        for file in files:
            try:
                contents = await file.read()
                filename = file.filename or "upload"
                document = await processor.ingest_upload(contents, filename)
                responses.append(self._build_ingestion_response(document))
            except Exception as exc:
                logger.error(
                    "Knowledge-base ingestion failed for %s: %s",
                    file.filename,
                    exc,
                    exc_info=True,
                )
                responses.append(
                    DocumentIngestionResponse(
                        success=False,
                        document_id="error",
                        filename=file.filename or "upload",
                        document_type="other",
                        status="failed",
                        error=str(exc),
                    )
                )

        return responses

    def _build_ingestion_response(self, document: Any) -> DocumentIngestionResponse:
        """Build the API response for a processed document."""
        return DocumentIngestionResponse(
            success=document.processing_status.value == "completed"
            and len(document.chunk_ids) > 0,
            document_id=document.document_id,
            filename=document.filename,
            document_type=document.document_type.value,
            status=document.processing_status.value,
            chunks_stored=len(document.chunk_ids),
            parent_chunks_stored=len(document.parent_chunk_ids),
            document_title=document.document_title,
            parsed_pages=document.parsed_pages,
            failed_pages=list(document.failed_pages or []),
            ocr_warnings=list(document.ocr_warnings or []),
            error=document.processing_error,
        )


class _SessionManagerAdapter:
    """Adapt the legacy function-based session access to the SessionManager protocol."""

    def get_or_create(self, session_id: str | None = None) -> tuple[ChatAgent, str]:
        """Delegate session access to the existing module function."""
        return get_or_create_agent(session_id)


chat_service = ChatRouteService(session_manager=_SessionManagerAdapter())
document_service = DocumentRouteService(
    processor_factory=create_document_processor,
)


async def handle_chat_message(message: str, session_id: str | None = None) -> ChatResponse:
    """Send a plain chat message to an agent session."""
    return await chat_service.handle_chat_message(message, session_id=session_id)


async def handle_chat_upload(
    message: str,
    files: list[UploadFile],
    session_id: str | None = None,
) -> ChatResponse:
    """Send uploaded files to an agent session."""
    return await chat_service.handle_chat_upload(
        message=message,
        files=files,
        session_id=session_id,
    )


async def handle_knowledge_base_ingestion(
    files: list[UploadFile],
) -> list[DocumentIngestionResponse]:
    """Ingest uploaded documents into the retrieval stack."""
    return await document_service.handle_knowledge_base_ingestion(files)
