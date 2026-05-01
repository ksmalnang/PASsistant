"""Retrieval node."""

import logging

from src.services.contracts import DocumentRetriever
from src.utils.state import AgentState, DocumentType
from src.utils.tools import VectorStoreTools

logger = logging.getLogger(__name__)


class RetrievalNode:
    """
    Retrieves relevant indexed document context from the retrieval stack.

    Delegates to the configured retrieval strategy (similarity, RRF, or reranker)
    and augments the state with hydrated parent/child retrieval context.
    """

    def __init__(self, retriever: DocumentRetriever | None = None):
        self.vector_tools = retriever or VectorStoreTools()

    async def run(self, state: AgentState) -> dict:
        """
        Retrieve relevant chunks for the current query.

        Args:
            state: Current agent state with retrieval_query

        Returns:
            State updates with retrieved chunks
        """
        if not state.requires_retrieval or not state.retrieval_query:
            return {}

        try:
            doc_type = self._resolve_document_type(state.current_intent)
            results = await self.vector_tools.search_similar(
                query=state.retrieval_query,
                document_type=doc_type,
                top_k=5,
                score_threshold=0.4,
            )
            logger.info(
                "Retrieved chunks",
                extra={"query": state.retrieval_query, "results": len(results)},
            )
            return {
                "retrieved_chunks": results,
                "requires_retrieval": False,
            }
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return {"error": f"Document retrieval failed: {exc}"}

    def _resolve_document_type(self, intent: str | None) -> DocumentType | None:
        """Resolve an optional document type filter from the current intent."""
        if intent == "query_student":
            return DocumentType.TRANSCRIPT
        return None
