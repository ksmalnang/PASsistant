"""Document indexing operations for the vector store."""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

from src.services.ingestion_health import IngestionHealthCheck
from src.utils.cache import RedisCache
from src.utils.state import DocumentUpload
from src.utils.tools.hierarchical_chunking import HierarchicalChunker
from src.utils.tools.parent_store import ParentChunkStore

logger = logging.getLogger(__name__)


class IndexingOperations:
    """Document chunk storage and deletion helpers."""

    MAX_UPSERT_PAYLOAD_BYTES = 28 * 1024 * 1024
    _UPSERT_BATCH_OVERHEAD_BYTES = 4096

    chunker: HierarchicalChunker
    embedding_model: str
    bm25_vector_name: str
    client: QdrantClient
    collection_name: str
    cache: RedisCache
    parent_store: ParentChunkStore
    ingestion_health_check: IngestionHealthCheck

    if TYPE_CHECKING:

        def _get_embeddings(self) -> Embeddings: ...
        def _ensure_collection_for_vector_size(self, vector_size: int) -> None: ...
        def ensure_collection(self) -> None: ...
        def _supports_bm25_vectors(self) -> bool: ...
        def _build_bm25_vector(self, text: str, *, is_query: bool = False) -> Any: ...

    async def store_document_chunks(
        self,
        document: DocumentUpload,
    ) -> list[str]:
        """
        Parse, chunk, embed, and store a document in the retrieval stack.

        Args:
            document: Document with extracted text

        Returns:
            List of stored child chunk IDs
        """
        if not document.extracted_text:
            raise ValueError("Document has no extracted text")

        structured_document = self.chunker.chunk_document(document)
        health_check = getattr(self, "ingestion_health_check", IngestionHealthCheck())
        report = health_check.validate(document, structured_document)
        document.document_title = structured_document.title
        document.parent_chunk_ids = [
            parent.parent_id for parent in structured_document.parent_chunks
        ]
        document.embedding_model = self.embedding_model
        document.ingestion_report = report.to_dict()
        document.quality_warning = self._summarize_ingestion_report(report)

        parent_records = [
            {
                "parent_id": parent.parent_id,
                "section_id": parent.section_id,
                "document_id": document.document_id,
                "filename": document.filename,
                "text": parent.text,
                "metadata": parent.metadata,
            }
            for parent in structured_document.parent_chunks
        ]

        child_texts = [chunk.text for chunk in structured_document.child_chunks]
        embeddings = (
            await self._get_embeddings().aembed_documents(child_texts) if child_texts else []
        )
        if embeddings:
            self._ensure_collection_for_vector_size(len(embeddings[0]))
        else:
            self.ensure_collection()
        supports_bm25_vectors = self._supports_bm25_vectors()

        chunk_ids: list[str] = []
        points: list[PointStruct] = []
        for index, (chunk, embedding) in enumerate(
            zip(structured_document.child_chunks, embeddings, strict=True)
        ):
            chunk_id = chunk.chunk_id or f"{document.document_id}_child_{index}"
            chunk_ids.append(chunk_id)
            point_id = self._build_point_id(document.document_id, chunk_id)

            vector: list[float] | dict[str, Any] = embedding
            if supports_bm25_vectors:
                vector = {"": embedding}
                bm25_vector = self._build_bm25_vector(chunk.text, is_query=False)
                if bm25_vector is not None:
                    vector[self.bm25_vector_name] = bm25_vector

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=self._build_child_payload(
                        document=document,
                        chunk=chunk,
                        chunk_id=chunk_id,
                        chunk_index=index,
                        total_chunks=len(structured_document.child_chunks),
                    ),
                )
            )

        self.delete_document_chunks(document.document_id)
        if points:
            try:
                self._upsert_points_in_batches(points)
                self.parent_store.put_many(parent_records)
            except Exception:
                logger.exception(
                    "Failed to store indexed document chunks; rolling back document index",
                    extra={"document_id": document.document_id},
                )
                self.delete_document_chunks(document.document_id)
                raise
        self.cache.delete_prefix("search:")

        logger.info(
            "Stored document chunks",
            extra={
                "document_id": document.document_id,
                "chunks_stored": len(structured_document.child_chunks),
                "parents_stored": len(structured_document.parent_chunks),
                "chunk_ids": chunk_ids,
            },
        )
        return chunk_ids

    def _upsert_points_in_batches(self, points: list[PointStruct]) -> None:
        """Send point upserts in payload-sized batches to stay below Qdrant limits."""
        batches = self._split_upsert_batches(points)
        if len(batches) > 1:
            logger.info(
                "Uploading indexed chunks in multiple Qdrant batches",
                extra={"batch_count": len(batches), "point_count": len(points)},
            )
        for batch_index, batch in enumerate(batches, start=1):
            logger.debug(
                "Upserting Qdrant batch",
                extra={
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "points_in_batch": len(batch),
                },
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

    def _split_upsert_batches(self, points: list[PointStruct]) -> list[list[PointStruct]]:
        """Partition points into request-sized batches based on JSON payload size."""
        if not points:
            return []

        max_bytes = getattr(self, "MAX_UPSERT_PAYLOAD_BYTES", self.MAX_UPSERT_PAYLOAD_BYTES)
        limit = max(int(max_bytes) - self._UPSERT_BATCH_OVERHEAD_BYTES, 1)
        batches: list[list[PointStruct]] = []
        current_batch: list[PointStruct] = []
        current_size = 0

        for point in points:
            point_size = self._estimate_point_payload_bytes(point)
            if point_size > limit:
                raise ValueError(
                    "A single indexed chunk is too large for Qdrant upsert payload limits. "
                    f"Estimated point size={point_size} bytes, limit={limit} bytes."
                )

            if current_batch and (current_size + point_size) > limit:
                batches.append(current_batch)
                current_batch = []
                current_size = 0

            current_batch.append(point)
            current_size += point_size

        if current_batch:
            batches.append(current_batch)
        return batches

    def _estimate_point_payload_bytes(self, point: PointStruct) -> int:
        """Estimate the serialized JSON size of one upsert point."""
        serialized = point.model_dump(mode="json", exclude_none=True)
        return len(json.dumps(serialized, ensure_ascii=False, separators=(",", ":")).encode())

    def _summarize_ingestion_report(self, report: Any) -> str | None:
        """Collapse ingestion issues into a short human-readable warning."""
        issues = list(getattr(report, "issues", []) or [])
        unrecognized = list(getattr(report, "unrecognized_headings", []) or [])
        if not issues and not unrecognized:
            return None

        parts = [f"{issue.code}: {issue.message}" for issue in issues[:3]]
        if unrecognized:
            parts.append(
                "UNRECOGNIZED_HEADINGS: "
                + ", ".join(unrecognized[:3])
            )
        return " | ".join(parts)

    def delete_document_chunks(self, document_id: str) -> None:
        """Delete all chunks for a specific document."""
        points_selector = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ],
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=points_selector,
        )
        self.parent_store.delete_document(document_id)
        self.cache.delete_prefix("search:")
        logger.info("Deleted chunks for document: %s", document_id)

    def _build_child_payload(
        self,
        document: DocumentUpload,
        chunk: Any,
        chunk_id: str,
        chunk_index: int,
        total_chunks: int,
    ) -> dict[str, Any]:
        """Build the Qdrant payload for an indexed child chunk."""
        payload = dict(chunk.metadata)
        payload.pop("bbox_2d", None)
        payload.pop("layout_details", None)
        payload.update(
            {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "document_id": document.document_id,
                "filename": document.filename,
                "document_type": document.document_type.value,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_at": document.uploaded_at.isoformat(),
            }
        )
        return payload

    def _build_point_id(self, document_id: str, chunk_id: str) -> str:
        """Map a logical chunk id to a deterministic UUID for Qdrant storage."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_id}"))
