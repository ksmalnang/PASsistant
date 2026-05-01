"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api import app
from src.api.models import ChatResponse, DependencyHealthResponse
from src.api.services import ChatRouteService
from src.utils.state import Citation, DocumentType, DocumentUpload, ProcessingStatus


def test_upload_documents_returns_created(monkeypatch):
    """Upload route should ingest documents and return HTTP 201."""

    async def fake_handle_knowledge_base_ingestion(files):
        assert len(files) == 1
        return [
            {
                "document_id": "doc-123",
                "filename": "knowledge-base.pdf",
                "document_type": "other",
                "status": "completed",
                "success": True,
                "chunks_stored": 2,
                "parent_chunks_stored": 1,
                "document_title": "Knowledge Base",
                "parsed_pages": None,
                "failed_pages": [],
                "ocr_warnings": [],
                "error": None,
            }
        ]

    monkeypatch.setattr(
        "src.api.routes.documents.handle_knowledge_base_ingestion",
        fake_handle_knowledge_base_ingestion,
    )

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"files": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json() == [
        {
            "success": True,
            "document_id": "doc-123",
            "filename": "knowledge-base.pdf",
            "document_type": "other",
            "status": "completed",
            "chunks_stored": 2,
            "parent_chunks_stored": 1,
            "document_title": "Knowledge Base",
            "parsed_pages": None,
            "failed_pages": [],
            "ocr_warnings": [],
            "error": None,
        }
    ]


def test_upload_documents_rejects_unsupported_mime_type():
    """Document upload route should reject unsupported MIME types."""

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"files": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported file type for 'notes.txt'. Only PDF, PNG, JPEG, GIF, and WEBP uploads are supported."
    }


def test_upload_documents_rejects_oversized_pdf(monkeypatch):
    """Document upload route should reject files above the configured PDF size limit."""

    monkeypatch.setattr("src.api.routes.documents.GLMOCRTool.MAX_PDF_BYTES", 4)

    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"files": ("large.pdf", b"12345", "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Uploaded file 'large.pdf' is too large. Maximum size for application/pdf is 0.0 MB."
    }


def test_ingest_knowledge_base_documents_success(monkeypatch):
    """Knowledge-base ingestion should return processing metadata."""

    class FakeProcessor:
        async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
            assert file_bytes == b"sample knowledge base content"
            assert filename == "knowledge-base.pdf"
            return DocumentUpload(
                document_id="doc-123",
                filename=filename,
                file_path="data/raw/doc-123.pdf",
                document_type=DocumentType.OTHER,
                mime_type="application/pdf",
                file_size=len(file_bytes),
                processing_status=ProcessingStatus.COMPLETED,
                chunk_ids=["child-1", "child-2"],
                parent_chunk_ids=["parent-1"],
                document_title="Knowledge Base",
            )

    monkeypatch.setattr("src.api.helpers.DocumentProcessingNode", FakeProcessor)

    client = TestClient(app)
    response = client.post(
        "/knowledge-base/ingest",
        files={"files": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "success": True,
            "document_id": "doc-123",
            "filename": "knowledge-base.pdf",
            "document_type": "other",
            "status": "completed",
            "chunks_stored": 2,
            "parent_chunks_stored": 1,
            "document_title": "Knowledge Base",
            "parsed_pages": None,
            "failed_pages": [],
            "ocr_warnings": [],
            "error": None,
        }
    ]


def test_ingest_knowledge_base_documents_failure(monkeypatch):
    """Knowledge-base ingestion should surface per-file processing failures."""

    class FakeProcessor:
        async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
            raise RuntimeError("vector store unavailable")

    monkeypatch.setattr("src.api.helpers.DocumentProcessingNode", FakeProcessor)

    client = TestClient(app)
    response = client.post(
        "/knowledge-base/ingest",
        files={"files": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "success": False,
            "document_id": "error",
            "filename": "knowledge-base.pdf",
            "document_type": "other",
            "status": "failed",
            "chunks_stored": 0,
            "parent_chunks_stored": 0,
            "document_title": None,
            "parsed_pages": None,
            "failed_pages": [],
            "ocr_warnings": [],
            "error": "vector store unavailable",
        }
    ]


def test_ingest_knowledge_base_documents_missing_files_returns_400():
    """Knowledge-base ingestion should reject requests without uploaded files."""

    client = TestClient(app)
    response = client.post("/knowledge-base/ingest")

    assert response.status_code == 422


def test_health_check_includes_dependency_statuses(monkeypatch):
    """Health route should expose Redis and Qdrant dependency states."""

    monkeypatch.setattr(
        "src.api.routes.health._check_redis_health",
        lambda: DependencyHealthResponse(status="healthy", detail="Redis responded to ping."),
    )
    monkeypatch.setattr(
        "src.api.routes.health._check_qdrant_health",
        lambda: DependencyHealthResponse(status="unhealthy", detail="connection refused"),
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "version": "0.1.0",
        "environment": "development",
        "redis": {
            "status": "healthy",
            "detail": "Redis responded to ping.",
        },
        "qdrant": {
            "status": "unhealthy",
            "detail": "connection refused",
        },
    }


def test_frontend_single_document_ingest_route_is_removed():
    """Removed frontend single-document ingest route should not be registered."""
    client = TestClient(app)
    response = client.post(
        "/documents/ingest",
        files={"file": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 404


def test_frontend_batch_document_ingest_route_is_removed():
    """Removed frontend batch ingest route should not be registered."""
    client = TestClient(app)
    response = client.post(
        "/documents/ingest/batch",
        files={"file": ("knowledge-base.pdf", b"sample knowledge base content", "application/pdf")},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_service_returns_actual_workflow_intent():
    class FakeAgent:
        async def chat_with_state(self, message: str, files=None):
            assert message == "apa saja Profil Lulusan dan deskripsinya"
            assert files is None
            return type(
                "State",
                (),
                {
                    "current_intent": "query_document",
                    "messages": [AIMessage(content="Profil lulusan ada empat.")],
                },
            )()

    class FakeSessionManager:
        def get_or_create(self, session_id=None):
            assert session_id == "session-1"
            return FakeAgent(), "session-1"

    service = ChatRouteService(session_manager=FakeSessionManager())

    response = await service.handle_chat_message(
        "apa saja Profil Lulusan dan deskripsinya",
        session_id="session-1",
    )

    assert response == ChatResponse(
        response="Profil lulusan ada empat.",
        session_id="session-1",
        intent="query_document",
        documents_processed=0,
    )


@pytest.mark.asyncio
async def test_chat_service_returns_workflow_citations():
    citation = Citation(
        id=1,
        document_id="doc-123",
        filename="kurikulum.pdf",
        section="3.1 Profil Lulusan",
        page=14,
        snippet="Profil lulusan dan deskripsinya",
    )

    class FakeAgent:
        async def chat_with_state(self, message: str, files=None):
            assert message == "apa saja Profil Lulusan dan deskripsinya"
            assert files is None
            return type(
                "State",
                (),
                {
                    "current_intent": "query_document",
                    "draft_response": "Profil lulusan ada empat.\n\nSources:\n[1] kurikulum.pdf",
                    "citations": [citation],
                    "messages": [],
                },
            )()

    class FakeSessionManager:
        def get_or_create(self, session_id=None):
            assert session_id == "session-1"
            return FakeAgent(), "session-1"

    service = ChatRouteService(session_manager=FakeSessionManager())

    response = await service.handle_chat_message(
        "apa saja Profil Lulusan dan deskripsinya",
        session_id="session-1",
    )

    assert response.citations == [citation]
    assert response.response.endswith("[1] kurikulum.pdf")
