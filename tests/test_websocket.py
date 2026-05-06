"""WebSocket streaming contract tests."""

from datetime import datetime

from fastapi.testclient import TestClient

from src.api import app
from src.api.models import ChatStreamEvent


def test_websocket_stream_emits_structured_events(monkeypatch):
    async def fake_stream(*, message: str, thread_id: str | None = None, last_event_id: str | None = None):
        del message, last_event_id
        yield ChatStreamEvent(
            event_id="run-1:1",
            event_type="run.started",
            thread_id=thread_id or "thread-1",
            run_id="run-1",
            timestamp=datetime.fromisoformat("2026-05-06T00:00:00+00:00"),
            sequence=1,
            data={"message": "accepted", "resume_supported": True},
        )
        yield ChatStreamEvent(
            event_id="run-1:2",
            event_type="message.delta",
            thread_id=thread_id or "thread-1",
            run_id="run-1",
            timestamp=datetime.fromisoformat("2026-05-06T00:00:01+00:00"),
            sequence=2,
            data={"text": "hello", "index": 5},
        )
        yield ChatStreamEvent(
            event_id="run-1:3",
            event_type="run.completed",
            thread_id=thread_id or "thread-1",
            run_id="run-1",
            timestamp=datetime.fromisoformat("2026-05-06T00:00:02+00:00"),
            sequence=3,
            data={"response": "hello", "intent": "general_chat", "documents_processed": 0, "citations": []},
        )

    monkeypatch.setattr("src.api.routes.websocket.chat_service.stream_chat_message", fake_stream)

    client = TestClient(app)
    with client.websocket_connect("/ws/thread-1") as websocket:
        websocket.send_json({"message": "hello"})
        started = websocket.receive_json()
        delta = websocket.receive_json()
        completed = websocket.receive_json()

    assert started["event_type"] == "run.started"
    assert delta["event_type"] == "message.delta"
    assert completed["event_type"] == "run.completed"
    assert completed["thread_id"] == "thread-1"
