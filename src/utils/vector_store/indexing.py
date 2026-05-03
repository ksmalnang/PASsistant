"""Document indexing operations for the vector store."""

import logging
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, PointStruct

from src.utils.cache import RedisCache
from src.utils.state import DocumentUpload
from src.utils.tools.hierarchical_chunking import HierarchicalChunker
from src.utils.tools.parent_store import ParentChunkStore

logger = logging.getLogger(__name__)


class IndexingOperations:
    """Document chunk storage and deletion helpers."""

    chunker: HierarchicalChunker
    embedding_model: str
    bm25_vector_name: str
    client: QdrantClient
    collection_name: str
    cache: RedisCache
    parent_store: ParentChunkStore

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
        document.document_title = structured_document.title
        document.parent_chunk_ids = [
            parent.parent_id for parent in structured_document.parent_chunks
        ]
        document.embedding_model = self.embedding_model

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
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
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
