"""LLM helper tests."""

from types import SimpleNamespace

from src.utils.nodes import llm as llm_module


def test_get_llm_passes_reasoning_extra_body(monkeypatch):
    """Configured LLM should include provider reasoning controls."""
    captured_kwargs = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            LLM_MODEL="deepseek/deepseek-v4-flash:exacto",
            LLM_REASONING_ENABLED=True,
            LLM_REASONING_EFFORT="high",
            LLM_REASONING_MAX_TOKENS=None,
            LLM_REASONING_EXCLUDE=True,
        ),
    )

    llm_module.get_llm()

    assert captured_kwargs["extra_body"] == {
        "reasoning": {
            "effort": "high",
            "exclude": True,
        }
    }


def test_get_llm_omits_reasoning_when_disabled(monkeypatch):
    """Reasoning can be disabled for providers that reject extra request fields."""
    captured_kwargs = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://example.test/v1",
            LLM_MODEL="provider/model",
            LLM_REASONING_ENABLED=False,
            LLM_REASONING_EFFORT="medium",
            LLM_REASONING_MAX_TOKENS=None,
            LLM_REASONING_EXCLUDE=True,
        ),
    )

    llm_module.get_llm()

    assert captured_kwargs["extra_body"] is None


def test_get_llm_uses_reasoning_max_tokens_when_configured(monkeypatch):
    """Max token budget should replace effort when configured."""
    captured_kwargs = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        llm_module,
        "get_settings",
        lambda: SimpleNamespace(
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://openrouter.ai/api/v1",
            LLM_MODEL="deepseek/deepseek-v4-flash:exacto",
            LLM_REASONING_ENABLED=True,
            LLM_REASONING_EFFORT="high",
            LLM_REASONING_MAX_TOKENS=2048,
            LLM_REASONING_EXCLUDE=True,
        ),
    )

    llm_module.get_llm()

    assert captured_kwargs["extra_body"] == {
        "reasoning": {
            "max_tokens": 2048,
            "exclude": True,
        }
    }
