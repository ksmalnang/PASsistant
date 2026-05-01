"""Intent routing node."""

from langchain_core.messages import HumanMessage

from src.services.contracts import LLMProvider
from src.services.intent import IntentClassifier
from src.utils.state import AgentState


class RouterNode:
    """
    Intent classification and routing node.

    Analyzes user message to determine:
    - upload_document: User wants to upload a document
    - query_student: User is asking about a student record or transcript-style data
    - query_document: User is asking about academic services or document content
    - manage_record: User wants to create a student record from provided text or documents
    - general_chat: General conversation
    """

    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        if classifier is not None:
            self._classifier = classifier
        elif llm_provider is not None:
            self._classifier = IntentClassifier(llm_provider=llm_provider)
        else:
            self._classifier = IntentClassifier()

    def run(self, state: AgentState) -> dict:
        """
        Classify intent and update state routing flags.

        Args:
            state: Current agent state

        Returns:
            State updates dict
        """
        if not state.messages:
            return {"current_intent": "general_chat"}

        last_message = state.messages[-1]
        if not isinstance(last_message, HumanMessage):
            return {}
        return self._classifier.classify(last_message.content, session_id=state.session_id)
