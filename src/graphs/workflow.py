"""
LangGraph workflow definition for the academic services and student records chatbot.

Defines the state graph with nodes and conditional edges,
compiling into a runnable LangGraph application.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.services.contracts import StateNode
from src.utils.nodes import (
    DocumentProcessingNode,
    ErrorHandlerNode,
    ResponseNode,
    RetrievalNode,
    RouterNode,
    StudentRecordNode,
)
from src.utils.state import AgentState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowNodes:
    """Concrete workflow node set used to assemble the graph."""

    router: StateNode
    document_processor: StateNode
    student_handler: StateNode
    retrieval: StateNode
    response: StateNode
    error_handler: StateNode


# =============================================================================
# Conditional Edge Functions
# =============================================================================


def route_by_intent(
    state: AgentState,
) -> Literal["process_document", "handle_student", "retrieve", "generate_response"]:
    """
    Route to appropriate node based on classified intent.

    Returns:
        Next node name
    """
    intent = state.current_intent

    if intent == "upload_document" or state.requires_upload:
        return "process_document"
    elif intent in ["query_student", "manage_record"]:
        return "handle_student"
    elif intent == "query_document" or state.requires_retrieval:
        return "retrieve"
    else:
        return "generate_response"


def check_processing_status(
    state: AgentState,
) -> Literal["handle_student", "generate_response"]:
    """
    Check if document processing succeeded and route accordingly.

    If documents were processed for student records, route to student handler.
    Otherwise, generate response about processing results.

    Returns:
        Next node name
    """
    if state.current_intent == "manage_record" and state.processed_documents:
        return "handle_student"
    return "generate_response"


def check_student_resolution(
    state: AgentState,
) -> Literal["retrieve", "generate_response"]:
    """
    Continue to retrieval when the student node could not answer directly.

    Returns:
        Next node name
    """
    if state.requires_retrieval and state.retrieval_query:
        return "retrieve"
    return "generate_response"


def check_retrieval_results(
    state: AgentState,
) -> Literal["generate_response", "fallback_response"]:
    """
    Check if retrieval produced results.

    Returns:
        Next node name
    """
    if state.retrieved_chunks:
        return "generate_response"
    return "fallback_response"


def check_errors(state: AgentState) -> str:
    """
    Check if there are errors that need handling.

    Returns:
        Next node name or END
    """
    if state.error and state.retry_count < 3:
        return "handle_error"
    return END


# =============================================================================
# Workflow Creation
# =============================================================================


def create_default_nodes() -> WorkflowNodes:
    """Create the default workflow node instances."""
    return WorkflowNodes(
        router=RouterNode(),
        document_processor=DocumentProcessingNode(),
        student_handler=StudentRecordNode(),
        retrieval=RetrievalNode(),
        response=ResponseNode(),
        error_handler=ErrorHandlerNode(),
    )


def create_workflow(nodes: WorkflowNodes | None = None) -> StateGraph:
    """
    Create the LangGraph workflow definition.

    Graph Structure:

                         ┌─────────────────┐
                         │   router_node   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              [upload]    [student record]  [service/general]
                    │             │             │
           ┌────────▼────────┐   │    ┌────────▼────────┐
           │ process_document │   │    │   retrieve      │
           └────────┬────────┘   │    └────────┬────────┘
                    │             │             │
                    │    ┌────────▼────────┐   │
                    │    │ handle_student  │   │
                    │    └────────┬────────┘   │
                    │             │            │
                    └──────┐   [needs docs]    │
                           │      │     │      │
                           │      │     └──────┘
                           │      │
                           │   ┌──▼───────┐
                           │   │ retrieve │
                           │   └──┬───────┘
                           │      │
                    ┌──────▼────────────▼──────▼──────┐
                    │       generate_response         │
                    └────────────────┬────────────────┘
                                     │
                            ┌────────▼────────┐
                            │  check_errors   │
                            └────────┬────────┘
                                     │
                              ┌──────┴──────┐
                              │             │
                         [has_error]    [no_error]
                              │             │
                    ┌─────────▼──────┐     │
                    │  handle_error  │     │
                    └────────┬───────┘     │
                             │             │
                             └──────┐      │
                                    │      │
                              ┌─────▼──────▼──────┐
                              │       END         │
                              └───────────────────┘

    Returns:
        StateGraph: Configured but uncompiled workflow
    """
    # Initialize workflow with state schema
    workflow = StateGraph(AgentState)

    # --- Initialize node instances ---
    resolved_nodes = nodes or create_default_nodes()

    # ====================================================================
    # Add Nodes
    # ====================================================================

    workflow.add_node("router_node", resolved_nodes.router.run)
    workflow.add_node("process_document", resolved_nodes.document_processor.run)
    workflow.add_node("handle_student", resolved_nodes.student_handler.run)
    workflow.add_node("retrieve", resolved_nodes.retrieval.run)
    workflow.add_node("generate_response", resolved_nodes.response.run)
    workflow.add_node("handle_error", resolved_nodes.error_handler.run)

    # Fallback response when no retrieval results
    workflow.add_node(
        "fallback_response",
        lambda state: {
            "messages": [
                {
                    "role": "assistant",
                    "content": "I couldn't find relevant information in the academic-service "
                    "documents or uploaded files. Could you try rephrasing your question or "
                    "upload a relevant document?",
                }
            ],
            "draft_response": "No relevant information found in academic-service documents.",
        },
    )

    # ====================================================================
    # Define Edges
    # ====================================================================

    # Entry point
    workflow.set_entry_point("router_node")

    # Router conditional edges
    workflow.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "process_document": "process_document",
            "handle_student": "handle_student",
            "retrieve": "retrieve",
            "generate_response": "generate_response",
        },
    )

    # Document processing -> check status
    workflow.add_conditional_edges(
        "process_document",
        check_processing_status,
        {
            "handle_student": "handle_student",
            "generate_response": "generate_response",
        },
    )

    # Student handler -> retrieval or response
    workflow.add_conditional_edges(
        "handle_student",
        check_student_resolution,
        {
            "retrieve": "retrieve",
            "generate_response": "generate_response",
        },
    )

    # Retrieval -> check results or response
    workflow.add_conditional_edges(
        "retrieve",
        check_retrieval_results,
        {
            "generate_response": "generate_response",
            "fallback_response": "fallback_response",
        },
    )

    # Fallback -> end
    workflow.add_edge("fallback_response", "generate_response")

    # Response -> error check
    workflow.add_conditional_edges(
        "generate_response",
        check_errors,
        {
            "handle_error": "handle_error",
            END: END,
        },
    )

    # Error handler -> end (prevent infinite loops)
    workflow.add_edge("handle_error", END)

    logger.info("Workflow graph created with %d nodes", len(workflow.nodes))
    return workflow


def compile_app(checkpointer=None, nodes: WorkflowNodes | None = None):
    """
    Compile the workflow into a runnable LangGraph application.

    Args:
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled application ready for invocation
    """
    workflow = create_workflow(nodes=nodes)

    if checkpointer is None:
        # Use in-memory checkpointer for development
        checkpointer = InMemorySaver()

    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_document"],  # Allow human review before processing
    )

    logger.info(
        "Application compiled with checkpointer: %s", type(checkpointer).__name__
    )
    return app
