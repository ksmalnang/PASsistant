"""Unit tests for batched query-variant retrieval."""

from types import SimpleNamespace
from typing import Any

import pytest

from src.utils.vector_store import VectorStoreTools


class _FakeEmbeddings:
    """Embeddings double that records batched and single query calls."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.single: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.single.append(text)
        return [0.1, 0.2]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


class _FakeCache:
    """Redis cache double backed by a dict."""

    def __init__(self) -> None:
        self.stored: dict[str, Any] = {}

    def get_json(self, key: str) -> Any | None:
        return self.stored.get(key)

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        self.stored[key] = value


class _FakeQdrant:
    """Minimal Qdrant client double covering the dense retrieval path."""

    def collection_exists(self, name: str) -> bool:
        return True

    def get_collection(self, name: str) -> Any:
        params = SimpleNamespace(vectors=SimpleNamespace(size=2), sparse_vectors=None)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def update_collection(self, **kwargs: Any) -> None:
        return None

    def query_points(self, **kwargs: Any) -> Any:
        point = SimpleNamespace(id="c1", score=0.9, payload={"chunk_id": "c1", "text": "beban studi"})
        return SimpleNamespace(points=[point])


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> tuple[VectorStoreTools, _FakeEmbeddings]:
    embeddings = _FakeEmbeddings()
    monkeypatch.setattr("src.utils.vector_store.tools.get_embeddings", lambda: embeddings)
    store = VectorStoreTools(client=_FakeQdrant(), cache=_FakeCache())
    store.retrieval_strategy = "similarity"
    return store, embeddings


async def test_unit_search_many_embeds_all_variants_once(
    tools: tuple[VectorStoreTools, _FakeEmbeddings],
) -> None:
    """Three query variants cost one embedding request and return three rankings."""
    store, embeddings = tools
    queries = ["pembayaran dpp", "syarat perwalian", "dpp perwalian"]

    results = await store.search_many(queries, top_k=3)

    assert embeddings.batches == [queries]
    assert embeddings.single == []
    assert len(results) == 3
    assert all(result[0]["chunk_id"] == "c1" for result in results)


async def test_unit_search_similar_still_embeds_one_query(
    tools: tuple[VectorStoreTools, _FakeEmbeddings],
) -> None:
    """Single-query retrieval keeps using the query embedding endpoint."""
    store, embeddings = tools

    results = await store.search_similar("beban sks", top_k=3)

    assert embeddings.single == ["beban sks"]
    assert embeddings.batches == []
    assert results[0]["chunk_id"] == "c1"
