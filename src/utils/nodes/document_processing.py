"""Document processing node."""

from src.services.contracts import (
    DocumentChunkIndexer,
    DocumentTextExtractor,
    DocumentUploadPreparer,
)
from src.services.document_processing import (
    DocumentIngestionService,
    DocumentProcessingService,
)
from src.utils.state import AgentState, DocumentUpload
from src.utils.tools import DocumentTools, GLMOCRTool, VectorStoreTools


class DocumentProcessingNode:
    """
    Handles document OCR, chunking, and retrieval indexing.

    Workflow:
    1. Save uploaded file
    2. Extract text using GLM-4 OCR
    3. Chunk and store in Qdrant
    4. Update document status
    """

    def __init__(
        self,
        text_extractor: DocumentTextExtractor | None = None,
        chunk_indexer: DocumentChunkIndexer | None = None,
        upload_preparer: DocumentUploadPreparer | None = None,
        processing_service: DocumentProcessingService | None = None,
        ingestion_service: DocumentIngestionService | None = None,
    ):
        self._processing_service = processing_service or DocumentProcessingService(
            text_extractor=text_extractor or GLMOCRTool(),
            chunk_indexer=chunk_indexer or VectorStoreTools(),
        )
        self._ingestion_service = ingestion_service or DocumentIngestionService(
            upload_preparer=upload_preparer or DocumentTools(),
            processor=self._processing_service,
        )

    async def run(self, state: AgentState) -> dict:
        """
        Process pending documents in state.

        Args:
            state: Current agent state with pending documents

        Returns:
            State updates with processed documents
        """
        if not state.pending_documents:
            return {}
        result = await self._processing_service.process_pending_documents(
            state.pending_documents
        )

        updates = {
            "processed_documents": state.processed_documents + result.processed_documents,
            "pending_documents": [],
        }
        if result.errors:
            updates["error"] = "; ".join(result.errors)
        return updates

    async def _process_document(self, document: DocumentUpload) -> None:
        """Run OCR and vector storage for a single document."""
        await self._processing_service.process_document(document)

    def prepare_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """
        Save an uploaded file and return local metadata only.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename

        Returns:
            DocumentUpload persisted to raw storage and ready for later processing
        """
        return self._ingestion_service.prepare_upload(file_bytes, filename)

    async def ingest_upload(self, file_bytes: bytes, filename: str) -> DocumentUpload:
        """
        Save and fully ingest an uploaded document.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename

        Returns:
            Fully processed DocumentUpload
        """
        return await self._ingestion_service.ingest_upload(file_bytes, filename)
