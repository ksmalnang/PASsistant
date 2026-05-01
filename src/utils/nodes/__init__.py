"""LangGraph node classes and supporting helpers for workflow assembly."""

from typing import TYPE_CHECKING

__all__ = [
    "DocumentProcessingNode",
    "ErrorHandlerNode",
    "ResponseNode",
    "RetrievalNode",
    "RouterNode",
    "StudentRecordNode",
]

if TYPE_CHECKING:
    from src.utils.nodes.document_processing import DocumentProcessingNode
    from src.utils.nodes.error_handler import ErrorHandlerNode
    from src.utils.nodes.response import ResponseNode
    from src.utils.nodes.retrieval import RetrievalNode
    from src.utils.nodes.router import RouterNode
    from src.utils.nodes.student_record import StudentRecordNode


def __getattr__(name: str):
    """Lazily import node classes to avoid package import cycles."""
    if name == "DocumentProcessingNode":
        from src.utils.nodes.document_processing import DocumentProcessingNode

        return DocumentProcessingNode
    if name == "ErrorHandlerNode":
        from src.utils.nodes.error_handler import ErrorHandlerNode

        return ErrorHandlerNode
    if name == "ResponseNode":
        from src.utils.nodes.response import ResponseNode

        return ResponseNode
    if name == "RetrievalNode":
        from src.utils.nodes.retrieval import RetrievalNode

        return RetrievalNode
    if name == "RouterNode":
        from src.utils.nodes.router import RouterNode

        return RouterNode
    if name == "StudentRecordNode":
        from src.utils.nodes.student_record import StudentRecordNode

        return StudentRecordNode
    raise AttributeError(name)
