"""BM25 vector helpers for Qdrant hybrid retrieval."""

import re
from typing import Any

from qdrant_client.http.models import SparseVector

BM25_MODEL_NAME = "Qdrant/bm25"
BM25_DISABLE_STEMMER = True


class BM25VectorOperations:
    """Build Qdrant-compatible BM25 vectors for hybrid retrieval."""

    _BM25_MODEL_NAME = BM25_MODEL_NAME
    _BM25_DISABLE_STEMMER = BM25_DISABLE_STEMMER
    _TOKEN_PATTERN: re.Pattern[str]
    _RRF_RANK_CONSTANT: int
    _bm25_text_embedder: Any

    def _get_bm25_cache_signature(self) -> str:
        """Return the BM25 settings that affect retrieval behavior."""
        return f"{self._BM25_MODEL_NAME}|disable_stemmer={self._BM25_DISABLE_STEMMER}"

    def _get_bm25_text_embedder(self) -> Any:
        """Create the BM25 encoder lazily to keep startup cheap."""
        embedder = getattr(self, "_bm25_text_embedder", None)
        if embedder is None:
            try:
                from fastembed.sparse import SparseTextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "FastEmbed BM25 support is not installed. Install qdrant-client[fastembed]."
                ) from exc

            # Avoid English-specific stemming assumptions for mixed-language corpora.
            embedder = SparseTextEmbedding(
                model_name=self._BM25_MODEL_NAME,
                disable_stemmer=self._BM25_DISABLE_STEMMER,
            )
            self._bm25_text_embedder = embedder
        return embedder

    def _build_bm25_vector(self, text: str, *, is_query: bool = False) -> SparseVector | None:
        """Encode text into a Qdrant-compatible BM25 vector."""
        if not self._TOKEN_PATTERN.findall(text):
            return None

        embedder = self._get_bm25_text_embedder()
        if is_query:
            embedding = next(embedder.query_embed(query=text), None)
        else:
            embedding = next(embedder.embed(documents=[text]), None)

        if embedding is None or len(embedding.indices) == 0:
            return None

        indices = (
            embedding.indices.tolist()
            if hasattr(embedding.indices, "tolist")
            else list(embedding.indices)
        )
        values = (
            embedding.values.tolist()
            if hasattr(embedding.values, "tolist")
            else list(embedding.values)
        )

        return SparseVector(indices=indices, values=values)

    def _rrf_weight(self, rank: int) -> float:
        """Return the Reciprocal Rank Fusion contribution for a 1-based rank."""
        return 1.0 / (self._RRF_RANK_CONSTANT + rank)


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
