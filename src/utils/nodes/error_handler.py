"""Error handling node."""

import logging

from langchain_core.messages import AIMessage

from src.utils.state import AgentState

logger = logging.getLogger(__name__)


class ErrorHandlerNode:
    """Handles errors and provides graceful fallbacks."""

    def run(self, state: AgentState) -> dict:
        """Handle any errors in state and provide recovery."""
        if not state.error:
            return {}

        error_msg = state.error
        logger.error("Handling error: %s", error_msg)

        if "OCR" in error_msg or "extraction" in error_msg.lower():
            response = "I had trouble reading that document. Please ensure the file is clear and try again."
        elif "Qdrant" in error_msg or "vector" in error_msg.lower():
            response = "I'm having trouble accessing the document database. Please try again in a moment."
        elif "student" in error_msg.lower():
            response = "I couldn't find the student record you're looking for. Could you provide more details?"
        else:
            response = "I encountered an unexpected issue. Please try again or rephrase your request."

        return {
            "messages": [AIMessage(content=response)],
            "error": None,
            "retry_count": state.retry_count + 1,
        }
