"""
Pytest configuration and shared fixtures.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agent import StudentRecordsAgent
from src.graphs.workflow import compile_app


@pytest.fixture
def memory_checkpointer():
    """Provide a fresh memory checkpointer."""
    return InMemorySaver()


@pytest.fixture
def compiled_test_app(memory_checkpointer):
    """Provide a compiled app for testing."""
    return compile_app(checkpointer=memory_checkpointer)


@pytest.fixture
async def test_agent():
    """Provide a configured test agent."""
    agent = StudentRecordsAgent(session_id="test-session-001")
    return agent
