"""Narrow interface contracts used across the application."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from src.utils.state import AgentState, DocumentType, DocumentUpload, OCRResult, StudentRecord


@runtime_checkable
class InvokableLLM(Protocol):
    """Minimal interface required from an LLM client."""

    def invoke(self, messages: Sequence[Any]) -> Any:
        """Run the LLM with the provided messages."""


LLMProvider = Callable[[], InvokableLLM | None]
RetrievalStrategy = Literal["similarity", "rrf", "reranker"]


@runtime_checkable
class DocumentUploadPreparer(Protocol):
    """Prepare uploaded files for downstream processing."""

    def save_upload(self, file_bytes: bytes, original_filename: str) -> DocumentUpload:
        """Persist upload bytes and return document metadata."""


@runtime_checkable
class DocumentTextExtractor(Protocol):
    """Extract text content from a stored document."""

    async def extract_text(
        self,
        file_path: str,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> OCRResult:
        """Return extracted text and a text quality score (0.0–1.0)."""


@runtime_checkable
class DocumentChunkIndexer(Protocol):
    """Store processed documents in the retrieval index."""

    async def store_document_chunks(self, document: DocumentUpload) -> list[str]:
        """Persist indexed chunks and return their identifiers."""


@runtime_checkable
class DocumentRetriever(Protocol):
    """Retrieve relevant document chunks for a query."""

    retrieval_strategy: RetrievalStrategy
    reranker_model: str | None
    reranker_candidate_multiplier: int

    async def search_similar(
        self,
        query: str,
        document_type: DocumentType | None = None,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Run retrieval using the configured similarity, RRF, or reranker strategy.

        `top_k` is the maximum number of hydrated parent results to return.
        `score_threshold` applies directly only to pure similarity retrieval.
        """


@runtime_checkable
class StudentRecordRepository(Protocol):
    """Persistence and lookup operations for student records."""

    def create_record(self, record: StudentRecord) -> StudentRecord:
        """Create and return a student record."""

    def get_record(self, student_id: str) -> StudentRecord | None:
        """Lookup a record by student id."""

    def find_by_email(self, email: str) -> StudentRecord | None:
        """Lookup a record by email."""


@runtime_checkable
class StudentTextExtractor(Protocol):
    """Fallback text extraction for student record creation."""

    def extract_from_text(self, text: str) -> dict[str, Any]:
        """Extract structured student data from raw text."""


@runtime_checkable
class DocumentProcessor(Protocol):
    """Public document-processor behavior used by the API layer."""

    def prepare_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Prepare a document for later processing."""

    async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """Prepare and fully ingest a document."""


@runtime_checkable
class ChatAgent(Protocol):
    """Minimal chat agent behavior required by route handlers."""

    session_id: str
    doc_processor: DocumentProcessor

    async def chat(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> str:
        """Return a chat response."""

    async def chat_with_state(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> AgentState:
        """Return the final workflow state for a chat turn."""

    def stream_chat(
        self,
        message: str,
        files: list[tuple[str, bytes]] | None = None,
    ) -> AsyncIterator[Any]:
        """Yield streaming chat updates."""


@runtime_checkable
class SessionManager(Protocol):
    """Resolve stateful chat agents for API requests."""

    def get_or_create(self, session_id: str | None = None) -> tuple[ChatAgent, str]:
        """Return an existing agent or create a new one."""


@runtime_checkable
class StateNode(Protocol):
    """Minimal workflow node interface."""

    def run(self, state: AgentState) -> Any:
        """Execute node logic for the provided state."""
