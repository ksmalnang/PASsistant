"""Unit tests for intent classification and routing (UT-006)."""

from types import SimpleNamespace

import pytest

from src.services.intent import IntentClassifier
from tests.doubles import FakeDecisionClient, FakeLLMProvider


def test_unit_006_intent_classifier_maps_known_messages() -> None:
    """Keyword-routed messages map to workflow intents without a Jev or LLM call."""
    provider = FakeLLMProvider(response="general_chat")
    decision = FakeDecisionClient(choice="general_chat", confidence=0.9)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    upload = classifier.classify("upload dokumen ini")
    assert upload["current_intent"] == "upload_document"
    assert upload["requires_upload"] is True

    academic = classifier.classify("graduation requirements")
    assert academic["current_intent"] == "query_document"
    assert academic["requires_retrieval"] is True
    assert academic["retrieval_query"] == "graduation requirements"

    student = classifier.classify("GPA saya berapa?")
    assert student["current_intent"] == "query_student"
    assert student["requires_retrieval"] is True

    # Keyword matches must not pay for a Jev or LLM round trip.
    assert decision.calls == 0
    assert provider.calls == 0

    general = classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert general["intent_confidence"] == 0.9
    assert decision.calls == 1
    assert provider.calls == 0


def test_unit_006_intent_classifier_falls_back_to_llm_on_low_confidence() -> None:
    """Low-confidence Jev classification falls back to LLM provider."""
    provider = FakeLLMProvider(response="general_chat")
    decision = FakeDecisionClient(choice="general_chat", confidence=0.3)
    classifier = IntentClassifier(llm_provider=provider, decision_client=decision)

    general = classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert decision.calls == 1
    assert provider.calls == 1
    assert general.get("intent_fallback") == "llm_fallback"


def test_unit_006_intent_classifier_llm_only_when_jev_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Jev is disabled, intent classification uses LLM directly."""
    monkeypatch.setattr(
        "src.services.intent.get_settings",
        lambda: SimpleNamespace(JEV_INTENT_ENABLED=False),
    )
    provider = FakeLLMProvider(response="general_chat")
    classifier = IntentClassifier(llm_provider=provider)

    general = classifier.classify("halo")
    assert general["current_intent"] == "general_chat"
    assert provider.calls == 1
