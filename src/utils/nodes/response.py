"""Response generation node."""

import logging

from langchain_core.messages import AIMessage

from src.services.contracts import LLMProvider
from src.services.response_generation import (
    ResponseContextBuilder,
    ResponseGenerationService,
)
from src.utils.state import AgentState

logger = logging.getLogger(__name__)


class ResponseNode:
    """
    Generates final responses from workflow state.

    Uses the configured LLM when available and falls back to deterministic
    service behavior when it is not. Constructs prompts based on:
    - Classified intent
    - Retrieved document chunks
    - Student record data
    - Conversation history
    """

    def __init__(
        self,
        service: ResponseGenerationService | None = None,
        context_builder: ResponseContextBuilder | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        if service is not None:
            self._service = service
        else:
            service_kwargs = {}
            if context_builder is not None:
                service_kwargs["context_builder"] = context_builder
            if llm_provider is not None:
                service_kwargs["llm_provider"] = llm_provider
            self._service = ResponseGenerationService(**service_kwargs)

    def run(self, state: AgentState) -> dict:
        """
        Generate response based on current state.

        Args:
            state: Current agent state with all context

        Returns:
            State updates with generated response
        """
        try:
            return self._service.generate(state)
        except Exception as exc:
            logger.error("Response generation failed: %s", exc)
            return {
                "draft_response": "I apologize, but I encountered an error processing your request. Please try again.",
                "messages": [
                    AIMessage(
                        content="I apologize, but I encountered an error. Please try again."
                    )
                ],
                "error": str(exc),
            }
