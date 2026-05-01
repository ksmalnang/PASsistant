"""Session management for API agents."""

from src.agent import StudentRecordsAgent
from src.services.session_registry import InMemorySessionManager

session_manager = InMemorySessionManager(agent_factory=StudentRecordsAgent)


def get_or_create_agent(session_id: str | None = None) -> tuple[StudentRecordsAgent, str]:
    """Get existing agent or create a new one for the session."""
    agent, active_session_id = session_manager.get_or_create(session_id)
    return agent, active_session_id
