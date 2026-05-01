"""Pydantic models for the API layer."""

from pydantic import BaseModel, Field

from src.utils.state import Citation


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(description="User message text")
    session_id: str | None = Field(
        default=None,
        description="Session identifier for continuity",
    )


class ChatResponse(BaseModel):
    """Chat response payload."""

    response: str = Field(description="Agent response text")
    session_id: str = Field(description="Session identifier")
    intent: str | None = Field(default=None, description="Classified intent")
    documents_processed: int = Field(
        default=0,
        description="Number of documents processed",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Document sources used for the answer",
    )


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    detail: str = Field(description="Human-readable error detail")


class DependencyHealthResponse(BaseModel):
    """Health status for an external dependency."""

    status: str = Field(description="Dependency state such as healthy, unhealthy, or disabled")
    detail: str | None = Field(
        default=None,
        description="Additional dependency status detail",
    )


class DocumentIngestionResponse(BaseModel):
    """Knowledge-base ingestion response."""

    success: bool = Field(default=True, description="Whether ingestion completed")
    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    document_type: str = Field(description="Detected document type")
    status: str = Field(description="Processing status")
    chunks_stored: int = Field(default=0, description="Number of vector chunks stored")
    parent_chunks_stored: int = Field(
        default=0,
        description="Number of parent chunks stored",
    )
    document_title: str | None = Field(
        default=None,
        description="Extracted or inferred document title",
    )
    parsed_pages: int | None = Field(
        default=None,
        description="Number of OCR pages parsed successfully",
    )
    failed_pages: list[int] = Field(
        default_factory=list,
        description="Zero-based OCR page indexes that failed",
    )
    ocr_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal OCR warnings",
    )
    error: str | None = Field(default=None, description="Processing error, if any")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current environment")
    redis: DependencyHealthResponse = Field(description="Redis connectivity status")
    qdrant: DependencyHealthResponse = Field(description="Qdrant connectivity status")
