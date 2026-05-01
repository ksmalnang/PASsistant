"""Session-manager implementations for route handlers."""

from __future__ import annotations

from collections.abc import Callable

from src.services.contracts import ChatAgent


class InMemorySessionManager:
    """In-memory agent registry keyed by session id."""

    def __init__(self, agent_factory: Callable[[str | None], ChatAgent]):
        self._agent_factory = agent_factory
        self._sessions: dict[str, ChatAgent] = {}

    def get_or_create(self, session_id: str | None = None) -> tuple[ChatAgent, str]:
        """Return an existing agent or create a new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id], session_id

        agent = self._agent_factory(session_id)
        self._sessions[agent.session_id] = agent
        return agent, agent.session_id
