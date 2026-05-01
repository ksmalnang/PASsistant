"""
Academic Services and Student Records Chatbot - Main Agent Entry Point

This module provides the primary interface for running the LangGraph agent.
It handles configuration, compilation, and execution of the chatbot workflow.

Usage:
    # Programmatic usage
    from src.agent import StudentRecordsAgent

    agent = StudentRecordsAgent()
    result = await agent.chat("Upload this transcript", files=[("transcript.pdf", bytes)])

    # LangGraph deployment (langgraph.json entry point)
    from src.agent import compiled_app
    # The `compiled_app` is the compiled LangGraph application for deployment
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.config import configure_logging, get_settings
from src.graphs.workflow import compile_app
from src.utils.nodes import DocumentProcessingNode
from src.utils.state import AgentState

configure_logging()
logger = logging.getLogger(__name__)


class StudentRecordsAgent:
    """
    High-level interface for the academic services and student records chatbot.

    Encapsulates the LangGraph workflow and provides clean methods
    for common operations like chatting and document uploads.

    Example:
        agent = StudentRecordsAgent()

        # Simple chat
        response = await agent.chat("What is John's GPA?")

        # Upload and process document
        response = await agent.chat(
            "Process this transcript",
            files=[("john_transcript.pdf", file_bytes)]
        )
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        app_factory: Callable[..., Any] = compile_app,
        document_processor: DocumentProcessingNode | None = None,
        checkpointer: InMemorySaver | None = None,
    ):
        """
        Initialize the agent.

        Args:
            session_id: Optional session identifier for persistence
        """
        self.settings = get_settings()
        self.session_id = session_id or str(uuid.uuid4())
        self.checkpointer = checkpointer or InMemorySaver()
        self.app = app_factory(checkpointer=self.checkpointer)
        self.doc_processor = document_processor or DocumentProcessingNode()

        logger.info("Agent initialized with session: %s", self.session_id)

    async def chat(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]] = None,
    ) -> str:
        """
        Send a message to the agent with optional file uploads.

        Args:
            message: User message text
            files: Optional list of (filename, file_bytes) tuples to upload

        Returns:
            Agent response text
        """
        final_state = await self.chat_with_state(message, files=files)
        return self._extract_response_text(final_state)

    async def chat_with_state(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]] = None,
    ) -> AgentState:
        """
        Run one chat turn and return the final workflow state.

        Args:
            message: User message text
            files: Optional list of (filename, file_bytes) tuples to upload

        Returns:
            Final agent state after graph execution
        """
        state = self._build_initial_state(message, files)
        config = self._build_run_config()

        try:
            result = await self.app.ainvoke(
                state.model_dump(),
                config=config,
            )
            return AgentState(**result)

        except Exception as exc:
            logger.error("Agent execution failed: %s", exc, exc_info=True)
            return AgentState(
                session_id=self.session_id,
                messages=state.messages,
                error=str(exc),
                draft_response=f"I encountered an error: {str(exc)}. Please try again.",
            )

    async def stream_chat(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]] = None,
    ):
        """
        Stream agent responses for real-time updates.

        Yields intermediate state updates during graph execution.

        Args:
            message: User message text
            files: Optional list of (filename, file_bytes) tuples

        Yields:
            Dict with state updates at each step
        """
        state = self._build_initial_state(message, files)
        config = self._build_run_config()

        async for update in self.app.astream(
            state.model_dump(),
            config=config,
            stream_mode="updates",
        ):
            yield update

    def _build_initial_state(
        self,
        message: str,
        files: Optional[list[tuple[str, bytes]]],
    ) -> AgentState:
        """Build the initial graph state for a chat turn."""
        state = AgentState(session_id=self.session_id)
        self._attach_files(state, files)
        state.messages.append(HumanMessage(content=message))
        return state

    def _attach_files(
        self,
        state: AgentState,
        files: Optional[list[tuple[str, bytes]]],
    ) -> None:
        """Prepare uploaded files and attach them to state."""
        if not files:
            return

        for filename, file_bytes in files:
            document = self.doc_processor.prepare_upload(file_bytes, filename)
            state.pending_documents.append(document)
        logger.info("Prepared %s documents for upload", len(files))

    def _build_run_config(self) -> dict[str, Any]:
        """Build the runtime configuration for LangGraph execution."""
        return {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": 25,
        }

    def _extract_response_text(self, final_state: AgentState) -> str:
        """Resolve the assistant response from the final graph state."""
        if final_state.messages:
            last_message = final_state.messages[-1]
            return last_message.content
        return "I processed your request but have no response to provide."


# =============================================================================
# LangGraph Deployment Entry Point
# =============================================================================

# Pre-compiled application for LangGraph deployment
# This is imported by langgraph.json as the entry point
compiled_app = compile_app()


# =============================================================================
# CLI / Direct Execution
# =============================================================================


async def main():
    """Interactive CLI for testing the agent."""
    print("=" * 60)
    print("Academic Services and Student Records Chatbot")
    print("Type 'quit' to exit, 'upload <filepath>' to upload a document")
    print("=" * 60)

    agent = StudentRecordsAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower().startswith("upload "):
                filepath = user_input[7:].strip()
                path = Path(filepath)

                if not path.exists():
                    print(f"File not found: {filepath}")
                    continue

                with open(path, "rb") as f:
                    file_bytes = f.read()

                print(f"Uploading {path.name}...")
                response = await agent.chat(
                    "Process this document",
                    files=[(path.name, file_bytes)],
                )
            else:
                response = await agent.chat(user_input)

            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
