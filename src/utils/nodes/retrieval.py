"""Retrieval node."""

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from src.services.contracts import DocumentRetriever
from src.utils.nodes.llm import get_llm
from src.utils.state import AgentState, DocumentType
from src.utils.tools import VectorStoreTools

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT = """Given the user question below, extract a concise search query
that would best retrieve relevant academic policy documents. Focus on key terms,
regulations, and specific topics mentioned.

User question: {question}

Search query (Indonesian, concise, keyword-rich):"""


class RetrievalNode:
    """
    Retrieves relevant indexed document context from the retrieval stack.

    Delegates to the configured retrieval strategy (similarity, RRF, or reranker)
    and augments the state with hydrated parent/child retrieval context.
    """

    _STOPWORDS = {
        "apa",
        "yang",
        "akan",
        "jika",
        "saya",
        "kami",
        "dan",
        "atau",
        "di",
        "ke",
        "dari",
        "untuk",
        "dengan",
        "selama",
        "tanpa",
        "mengajukan",
        "mengikuti",
        "bagaimana",
        "adalah",
        "the",
        "is",
        "of",
        "to",
        "a",
        "an",
    }

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        llm_provider=get_llm,
    ):
        self.vector_tools = retriever or VectorStoreTools()
        self._llm_provider = llm_provider
        self._llm = None

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
            queries = self._build_queries(state)
            if not queries:
                return {
                    "retrieved_chunks": [],
                    "requires_retrieval": False,
                    "response_confidence": 0.0,
                    "retrieval_warning": (
                        "The retriever did not receive a usable query. "
                        "Ask the user to restate the question."
                    ),
                }
            merged_results: list[dict[str, Any]] = []
            for query in queries:
                query_results = await self.vector_tools.search_similar(
                    query=query,
                    document_type=doc_type,
                    top_k=5,
                    score_threshold=0.4,
                )
                merged_results = self._merge_results(merged_results, query_results, query)

            confidence, warning = self._score_retrieval_confidence(
                query=queries[0],
                results=merged_results,
            )
            logger.info(
                "Retrieved chunks",
                extra={
                    "query": state.retrieval_query,
                    "queries": queries,
                    "results": len(merged_results),
                    "confidence": confidence,
                },
            )
            return {
                "retrieved_chunks": merged_results[:5],
                "requires_retrieval": False,
                "response_confidence": confidence,
                "retrieval_warning": warning,
            }
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return {"error": f"Document retrieval failed: {exc}"}

    def _resolve_document_type(self, intent: str | None) -> DocumentType | None:
        """Resolve an optional document type filter from the current intent."""
        if intent == "query_student":
            return DocumentType.TRANSCRIPT
        return None

    def _build_queries(self, state: AgentState) -> list[str]:
        """Build original, rewritten, and synonym-expanded retrieval queries."""
        original = (state.retrieval_query or "").strip()
        if not original:
            return []
        if state.current_intent != "query_document":
            return [original]

        queries = [original]
        rewritten = self._rewrite_query(original)
        if rewritten and rewritten.lower() != original.lower():
            queries.append(rewritten)

        expanded = self._expand_query(rewritten or original)
        if expanded and all(expanded.lower() != query.lower() for query in queries):
            queries.append(expanded)
        return queries[:3]

    def _rewrite_query(self, question: str) -> str:
        """Compress a conversational question into a keyword-rich retrieval query."""
        llm = self._get_llm()
        if llm is not None:
            try:
                response = llm.invoke(
                    [HumanMessage(content=QUERY_REWRITE_PROMPT.format(question=question))]
                )
                rewritten = str(response.content).strip()
                if rewritten:
                    return rewritten.splitlines()[0].strip()
            except Exception as exc:
                logger.warning("Query rewrite failed, falling back to heuristics: %s", exc)
        return self._extract_keywords(question)

    def _expand_query(self, query: str) -> str:
        """Add a small synonym layer for Indonesian academic-policy retrieval."""
        expansions = {
            "absen": "ketidakhadiran",
            "tidak aktif": "nonaktif",
            "cuti": "cuti akademik",
            "masa studi": "akhir masa studi",
            "drop out": "dikeluarkan",
            "semester": "semester berturut-turut",
        }
        parts = [query.strip()]
        lowered = query.lower()
        for source, target in expansions.items():
            if source in lowered and target not in lowered:
                parts.append(target)
        return " ".join(part for part in parts if part).strip()

    def _extract_keywords(self, text: str) -> str:
        """Heuristic fallback for query rewriting when no LLM is configured."""
        tokens = re.findall(r"\b[\w-]+\b", text.lower())
        kept: list[str] = []
        for token in tokens:
            if token in self._STOPWORDS or len(token) <= 2:
                continue
            if token not in kept:
                kept.append(token)
        return " ".join(kept[:12]) or text.strip()

    def _merge_results(
        self,
        current: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        query_variant: str,
    ) -> list[dict[str, Any]]:
        """Merge per-query retrieval results by hydrated parent id."""
        merged: dict[str, dict[str, Any]] = {
            self._result_key(result): self._tag_result(result, result.get("matched_query"))
            for result in current
        }
        for result in incoming:
            key = self._result_key(result)
            tagged = self._tag_result(result, query_variant)
            if key not in merged:
                merged[key] = tagged
                continue

            existing = merged[key]
            if float(tagged.get("score") or 0.0) > float(existing.get("score") or 0.0):
                existing.update(
                    {
                        "score": tagged.get("score"),
                        "vector_score": tagged.get("vector_score", existing.get("vector_score")),
                        "bm25_score": tagged.get("bm25_score", existing.get("bm25_score")),
                        "rrf_score": tagged.get("rrf_score", existing.get("rrf_score")),
                        "matched_query": query_variant,
                    }
                )
            existing_children = {
                str(child.get("chunk_id") or child.get("text") or "")
                for child in existing.get("matched_children", [])
                if isinstance(child, dict)
            }
            for child in tagged.get("matched_children", []):
                child_key = str(child.get("chunk_id") or child.get("text") or "")
                if child_key and child_key not in existing_children:
                    existing.setdefault("matched_children", []).append(child)
                    existing_children.add(child_key)

        return sorted(
            merged.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

    def _score_retrieval_confidence(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> tuple[float, str | None]:
        """Estimate whether the retrieved context is likely on-topic."""
        if not results:
            return 0.0, (
                "The retriever did not find relevant indexed context. "
                "Acknowledge the gap and suggest uploading the specific document or "
                "contacting the academic office."
            )

        query_terms = set(self._extract_keywords(query).split())
        scored_results: list[dict[str, Any]] = []
        overlap_hits = 0
        strong_hits = 0
        for result in results[:5]:
            evidence_text = " ".join(
                [
                    str(result.get("breadcrumb") or ""),
                    str(result.get("text") or ""),
                    " ".join(
                        str(child.get("text") or "")
                        for child in result.get("matched_children", [])
                        if isinstance(child, dict)
                    ),
                ]
            ).lower()
            overlap_terms = sorted(term for term in query_terms if term and term in evidence_text)
            overlap_ratio = len(overlap_terms) / max(1, len(query_terms))
            annotated = dict(result)
            annotated["query_overlap_terms"] = overlap_terms
            annotated["query_overlap_ratio"] = overlap_ratio
            scored_results.append(annotated)
            if overlap_terms:
                overlap_hits += 1
            if float(result.get("score") or 0.0) >= 0.55:
                strong_hits += 1

        results[:] = scored_results
        top_score = float(scored_results[0].get("score") or 0.0)
        second_score = (
            float(scored_results[1].get("score") or 0.0)
            if len(scored_results) > 1
            else max(top_score - 0.05, 0.0)
        )
        score_component = min(max((top_score - 0.35) / 0.45, 0.0), 1.0)
        margin_component = min(max((top_score - second_score + 0.05) / 0.25, 0.0), 1.0)
        overlap_component = overlap_hits / max(1, min(len(scored_results), 3))
        hit_component = min(strong_hits / 2, 1.0)
        confidence = round(
            (0.45 * score_component)
            + (0.20 * margin_component)
            + (0.20 * overlap_component)
            + (0.15 * hit_component),
            3,
        )
        if confidence >= 0.45:
            return confidence, None
        return confidence, (
            "The retrieved excerpts may not directly answer the question. "
            "If the context looks off-topic, say so explicitly and suggest uploading the "
            "specific academic policy document or contacting the academic office."
        )

    def _result_key(self, result: dict[str, Any]) -> str:
        """Build a stable dedupe key for hydrated parent results."""
        return str(
            result.get("parent_id")
            or f"{result.get('document_id', '')}:{result.get('section_id', '')}:{result.get('chunk_id', '')}"
        )

    def _tag_result(self, result: dict[str, Any], query_variant: str | None) -> dict[str, Any]:
        """Attach the query variant that produced a result."""
        tagged = dict(result)
        if query_variant:
            tagged["matched_query"] = query_variant
        return tagged

    def _get_llm(self):
        """Resolve the current LLM lazily."""
        if self._llm is None:
            self._llm = self._llm_provider()
        return self._llm
