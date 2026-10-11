"""Lifespan wiring tests for the API app."""

import pytest
from fastapi import FastAPI

from src.api import lifespan
from src.clients import JevDecisionClient
from src.utils.vector_store import VectorStoreTools


async def test_api_lifespan_01_shutdown_closes_decision_client(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App shutdown on the event loop closes the decision client built during a request."""
    monkeypatch.setattr(VectorStoreTools, "warm_up", lambda self: None)
    client = JevDecisionClient(api_key="test-key")
    monkeypatch.setattr("src.clients.decision._decision_client", client)

    async with lifespan(app):
        pass

    assert client._client.is_closed
