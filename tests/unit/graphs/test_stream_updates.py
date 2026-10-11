"""Unit tests for the agent streaming update contract."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from src.agent import PASsistantAgent


class _EchoingGraph:
    """Compiled-graph double that repeats streamed text as a state message.

    Mirrors LangGraph messages stream mode, where the AIMessage the response
    node returns arrives in addition to the incremental chunks.
    """

    def __init__(self) -> None:
        self._meta = {"langgraph_node": "generate_response"}

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[tuple[str, Any]]:
        yield "messages", (AIMessageChunk(content="Halo "), self._meta)
        yield "messages", (AIMessageChunk(content="dunia"), self._meta)
        yield "messages", (AIMessage(content="Halo dunia"), self._meta)
        yield "updates", {"generate_response": {"draft_response": "Halo dunia"}}

    async def aget_state(self, config: Any) -> Any:
        return SimpleNamespace(values={"draft_response": "Halo dunia"})


async def test_unit_stream_chat_emits_each_chunk_once() -> None:
    """Only incremental chunks become tokens; the echoed state message is dropped."""
    agent = PASsistantAgent(
        app_factory=lambda checkpointer: _EchoingGraph(),
        checkpointer=InMemorySaver(),
    )

    updates = [update async for update in agent.stream_chat("apa itu KRS?")]

    assert "".join(update.text for update in updates if update.kind == "token") == "Halo dunia"
    assert updates[-1].kind == "final"
