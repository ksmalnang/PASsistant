"""
Unit tests for the LangGraph workflow.

Tests core graph functionality including routing,
document processing, and response generation.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.graphs.workflow import check_student_resolution
from src.services.document_processing import DocumentProcessingService
from src.services.response_generation import CitationBuilder, ResponseContextBuilder
from src.utils.nodes import ResponseNode, RouterNode
from src.utils.state import AgentState, DocumentUpload, OCRResult, ProcessingStatus


class TestRouterNode:
    """Test intent classification routing."""

    def test_router_detects_upload_intent(self):
        """Router should classify upload-related messages."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="I want to upload a transcript PDF")]
        )

        result = router.run(state)

        assert result["current_intent"] == "upload_document"
        assert result["requires_upload"] is True

    def test_router_detects_query_intent(self):
        """Router should classify student query messages."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="What is my GPA?")]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_student"
        assert result["requires_retrieval"] is True

    def test_router_detects_academic_service_query(self):
        """Router should classify academic service questions for retrieval."""
        router = RouterNode()
        state = AgentState(
            messages=[
                HumanMessage(
                    content="How do I request an enrollment verification letter?"
                )
            ]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_document"
        assert result["requires_retrieval"] is True

    def test_router_detects_curriculum_document_query_in_indonesian(self):
        """Router should classify curriculum-document questions for retrieval."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="apa saja Profil Lulusan dan deskripsinya")]
        )

        result = router.run(state)

        assert result["current_intent"] == "query_document"
        assert result["requires_retrieval"] is True

    def test_router_detects_general_chat(self):
        """Router should classify general conversation."""
        router = RouterNode()
        state = AgentState(
            messages=[HumanMessage(content="Hello, how are you?")]
        )

        result = router.run(state)

        assert result["current_intent"] == "general_chat"


class TestResponseNode:
    """Test response generation."""

    def test_citation_builder_uses_child_source_locations(self):
        """Citation metadata should prefer the matched child source location."""
        builder = CitationBuilder()

        citations = builder.build(
            [
                {
                    "document_id": "doc-123",
                    "filename": "kurikulum.pdf",
                    "doc_title": "Kurikulum Teknik Informatika",
                    "breadcrumb": "3.1 Profil Lulusan",
                    "score": 0.97,
                    "chunk_id": "parent-chunk",
                    "parent_id": "doc-123:bab_3.pasal_3_1",
                    "text": "Parent text",
                    "source_locations": [
                        {"page": 9, "bbox_2d": [1.0, 2.0, 3.0, 4.0]}
                    ],
                    "matched_children": [
                        {
                            "chunk_id": "child-chunk",
                            "text": "Profil lulusan dan deskripsinya",
                            "source_locations": [
                                {"page": 13, "bbox_2d": [10.0, 20.0, 100.0, 50.0]}
                            ],
                        }
                    ],
                }
            ]
        )

        assert len(citations) == 1
        assert citations[0].id == 1
        assert citations[0].filename == "kurikulum.pdf"
        assert citations[0].title == "Kurikulum Teknik Informatika"
        assert citations[0].section == "3.1 Profil Lulusan"
        assert citations[0].page == 14
        assert citations[0].chunk_id == "child-chunk"
        assert citations[0].source_locations == [
            {"page": 13, "bbox_2d": [10.0, 20.0, 100.0, 50.0]}
        ]

    def test_citation_builder_deduplicates_parent_chunks(self):
        """Multiple child hits under one parent should produce one citation."""
        builder = CitationBuilder()

        citations = builder.build(
            [
                {
                    "filename": "policy.pdf",
                    "parent_id": "doc-1:section-1",
                    "section_id": "section-1",
                    "score": 0.9,
                    "text": "First match",
                },
                {
                    "filename": "policy.pdf",
                    "parent_id": "doc-1:section-1",
                    "section_id": "section-1",
                    "score": 0.8,
                    "text": "Duplicate parent",
                },
            ]
        )

        assert len(citations) == 1

    def test_response_with_context(self):
        """Response should incorporate retrieved context."""
        response_node = ResponseNode()
        state = AgentState(
            messages=[
                HumanMessage(content="What courses did John take?")
            ],
            retrieved_chunks=[
                {
                    "text": "John enrolled in CS101, MATH201, and PHYS101 in Fall 2024.",
                    "score": 0.95,
                    "filename": "transcript.pdf",
                }
            ],
            current_intent="query_document",
        )

        result = response_node.run(state)

        assert "draft_response" in result
        assert len(result["messages"]) > 0
        assert result["citations"][0].filename == "transcript.pdf"
        assert "Sources:" in result["draft_response"]

    def test_response_without_context(self):
        """Response should handle missing context gracefully."""
        response_node = ResponseNode()
        state = AgentState(
            messages=[HumanMessage(content="Hello!")],
            current_intent="general_chat",
        )

        result = response_node.run(state)

        assert "draft_response" in result
        assert len(result["messages"]) > 0

    def test_response_context_preserves_table_rows_beyond_short_preview(self):
        """Table-heavy retrieval context should keep later rows such as PL2/PL3."""
        builder = ResponseContextBuilder()
        large_table_text = (
            "3.1 Profil Lulusan\n\n"
            + ("Pendahuluan. " * 80)
            + "| PL1 | Professional bidang Informatika | Deskripsi PL1 |\n"
            + "| PL2 | Technopreneur | Deskripsi PL2 |\n"
            + "| PL3 | Akademisi | Deskripsi PL3 |\n"
            + "| PL4 | Global citizen | Deskripsi PL4 |\n"
        )

        context_lines = builder._build_retrieval_context(
            [
                {
                    "filename": "kurikulum.pdf",
                    "breadcrumb": "3.1 Profil Lulusan",
                    "score": 0.97,
                    "text": large_table_text,
                    "matched_children": [{"text": "Profil lulusan dan deskripsinya"}],
                }
            ]
        )

        assert "PL3" in context_lines[0]


class TestAgentState:
    """Test state schema operations."""

    def test_state_initialization(self):
        """State should initialize with defaults."""
        state = AgentState()

        assert state.turn_count == 0
        assert state.current_intent is None
        assert len(state.pending_documents) == 0
        assert len(state.processed_documents) == 0

    def test_state_with_document(self):
        """State should track pending documents."""
        doc = DocumentUpload(
            document_id="test-doc-001",
            filename="transcript.pdf",
            file_path="/tmp/transcript.pdf",
            mime_type="application/pdf",
            file_size=1024,
        )

        state = AgentState(
            pending_documents=[doc],
            current_intent="upload_document",
        )

        assert len(state.pending_documents) == 1
        assert state.pending_documents[0].document_id == "test-doc-001"


class TestDocumentProcessingService:
    """Test document processing metadata behavior."""

    @pytest.mark.asyncio
    async def test_partial_ocr_completes_with_warnings(self):
        """Partial OCR should complete while preserving page status metadata."""
        document = DocumentUpload(
            document_id="doc-partial",
            filename="partial.pdf",
            file_path="/tmp/partial.pdf",
            mime_type="application/pdf",
            file_size=1024,
        )

        class FakeExtractor:
            async def extract_text(self, file_path, document_type):
                assert file_path == "/tmp/partial.pdf"
                return OCRResult(
                    text="page zero text",
                    text_quality_score=0.95,
                    num_pages=2,
                    layout_details=[
                        [{"content": "page zero text", "bbox_2d": [1, 2, 3, 4]}],
                        [],
                    ],
                    parsed_pages=1,
                    failed_pages=[1],
                    page_results=[
                        {"page_index": 0, "status": "success", "text_length": 14},
                        {"page_index": 1, "status": "failed", "error": "timeout"},
                    ],
                    ocr_warnings=["OCR parsed 1 of 2 pages successfully."],
                )

        class FakeIndexer:
            async def store_document_chunks(self, indexed_document):
                assert indexed_document.parsed_pages == 1
                assert indexed_document.failed_pages == [1]
                return ["chunk-1"]

        service = DocumentProcessingService(
            text_extractor=FakeExtractor(),
            chunk_indexer=FakeIndexer(),
        )

        await service.process_document(document)

        assert document.processing_status == ProcessingStatus.COMPLETED
        assert document.processing_error == "OCR parsed 1 of 2 pages successfully."
        assert document.parsed_pages == 1
        assert document.failed_pages == [1]
        assert document.ocr_page_status[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_processing_fails_when_no_vector_chunks_are_stored(self):
        """Completed OCR without vector chunks should not look like successful ingest."""
        document = DocumentUpload(
            document_id="doc-empty-index",
            filename="empty-index.pdf",
            file_path="/tmp/empty-index.pdf",
            mime_type="application/pdf",
            file_size=1024,
        )

        class FakeExtractor:
            async def extract_text(self, file_path, document_type):
                return OCRResult(
                    text="valid OCR text",
                    text_quality_score=0.95,
                    num_pages=1,
                    layout_details=[[{"content": "valid OCR text"}]],
                    parsed_pages=1,
                )

        class EmptyIndexer:
            async def store_document_chunks(self, indexed_document):
                assert indexed_document.extracted_text == "valid OCR text"
                return []

        service = DocumentProcessingService(
            text_extractor=FakeExtractor(),
            chunk_indexer=EmptyIndexer(),
        )

        with pytest.raises(RuntimeError, match="no vector chunks were stored"):
            await service.process_document(document)

        assert document.processing_status == ProcessingStatus.FAILED
        assert document.processing_error == (
            "Document ingestion completed OCR but no vector chunks were stored"
        )


class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    def test_student_resolution_routes_to_retrieval_when_needed(self):
        """Student handler should forward unresolved record questions to retrieval."""
        state = AgentState(
            current_intent="query_student",
            requires_retrieval=True,
            retrieval_query="What is my GPA?",
        )

        assert check_student_resolution(state) == "retrieve"

    @pytest.mark.asyncio
    async def test_full_chat_flow(self, compiled_test_app):
        """Test complete message processing through the graph."""
        initial_state = AgentState(
            messages=[HumanMessage(content="Hi there!")],
            session_id="test-integration",
        )

        config = {"configurable": {"thread_id": "test-integration"}}

        result = await compiled_test_app.ainvoke(
            initial_state.model_dump(),
            config=config,
        )

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1  # User + assistant messages
