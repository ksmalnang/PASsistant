"""Qdrant collection lifecycle and client-compatibility helpers."""

import logging
from typing import Any

from qdrant_client.http.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

logger = logging.getLogger(__name__)


class CollectionOperations:
    """Collection lifecycle helpers shared by vector-store operations."""

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist and detect BM25-vector support."""
        self._ensure_collection_for_vector_size(self.vector_size)

    def _ensure_collection_for_vector_size(self, vector_size: int) -> None:
        """Ensure the collection exists and matches the requested dense vector size."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config=self._build_bm25_vectors_config(),
            )
            self.bm25_vectors_enabled = True
            self.vector_size = vector_size
            logger.info(
                "Created Qdrant collection: %s",
                self.collection_name,
                extra={"vector_size": vector_size},
            )
            return

        info = self.client.get_collection(self.collection_name)
        current_vector_size = self._get_collection_vector_size(info)
        if current_vector_size is not None and current_vector_size != vector_size:
            points_count = getattr(info, "points_count", 0) or 0
            if points_count == 0:
                logger.warning(
                    "Recreating empty Qdrant collection %s due to vector size mismatch",
                    self.collection_name,
                    extra={
                        "configured_vector_size": current_vector_size,
                        "embedding_vector_size": vector_size,
                    },
                )
                self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                    sparse_vectors_config=self._build_bm25_vectors_config(),
                )
                self.bm25_vectors_enabled = True
                self.vector_size = vector_size
                return
            raise RuntimeError(
                "Qdrant collection vector size mismatch: "
                f"collection={current_vector_size}, embeddings={vector_size}. "
                f"Recreate collection '{self.collection_name}' or use a matching embedding model."
            )

        self.bm25_vectors_enabled = self._collection_has_bm25_vectors()
        self.vector_size = current_vector_size or vector_size
        if self.bm25_vectors_enabled:
            return

        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                sparse_vectors_config=self._build_bm25_vectors_config(),
            )
        except Exception as exc:
            self.bm25_vectors_enabled = False
            if self.retrieval_strategy == "rrf" and not self._rrf_warning_emitted:
                logger.warning(
                    "Collection %s does not support BM25 vectors yet; "
                    "RRF will fall back to dense-only retrieval until the collection is recreated "
                    "or reindexed. Update error: %s",
                    self.collection_name,
                    exc,
                )
                self._rrf_warning_emitted = True
            return

        self.bm25_vectors_enabled = self._collection_has_bm25_vectors()

    def _query_points(self, **kwargs: Any) -> list[Any]:
        """Normalize Qdrant query responses across client versions and test doubles."""
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(**kwargs)
            if isinstance(response, list):
                return response
            points = getattr(response, "points", None)
            if points is not None:
                return list(points)
            result = getattr(response, "result", None)
            if result is not None:
                return list(result)
            raise TypeError(f"Unsupported query_points response type: {type(response)!r}")

        if kwargs.get("using") or kwargs.get("prefetch"):
            raise RuntimeError("Hybrid retrieval requires a qdrant-client with query_points support.")

        return list(
            self.client.search(
                collection_name=kwargs["collection_name"],
                query_vector=kwargs["query"],
                query_filter=kwargs.get("query_filter"),
                limit=kwargs["limit"],
                score_threshold=kwargs.get("score_threshold"),
            )
        )

    def _build_bm25_vectors_config(self) -> dict[str, SparseVectorParams]:
        """Return the BM25 vector configuration used for hybrid retrieval."""
        return {
            self.bm25_vector_name: SparseVectorParams(
                modifier=Modifier.IDF,
            )
        }

    def _collection_has_bm25_vectors(self) -> bool:
        """Check whether the current collection exposes the configured BM25 vector field."""
        info = self.client.get_collection(self.collection_name)
        sparse_vectors = getattr(info.config.params, "sparse_vectors", None)
        if sparse_vectors is None:
            return False
        return self.bm25_vector_name in sparse_vectors

    def _get_collection_vector_size(self, info: Any) -> int | None:
        """Read the dense vector size from a Qdrant collection info object."""
        vectors = getattr(info.config.params, "vectors", None)
        if vectors is None:
            return None
        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            if first is not None and getattr(first, "size", None) is not None:
                return int(first.size)
        return None

    def _supports_bm25_vectors(self) -> bool:
        """Return whether BM25 vectors are available in the current collection."""
        return self.bm25_vectors_enabled is True
