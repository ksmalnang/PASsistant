"""Shared LLM helpers for node implementations."""

from langchain_openai import ChatOpenAI

from src.config import get_settings


def _build_reasoning_extra_body(settings) -> dict | None:
    """Build provider reasoning controls for OpenAI-compatible requests."""
    if not settings.LLM_REASONING_ENABLED:
        return None

    reasoning = {"exclude": settings.LLM_REASONING_EXCLUDE}
    if settings.LLM_REASONING_MAX_TOKENS is not None:
        reasoning["max_tokens"] = settings.LLM_REASONING_MAX_TOKENS
    else:
        reasoning["effort"] = settings.LLM_REASONING_EFFORT

    return {"reasoning": reasoning}


def get_llm():
    """Get configured LLM instance."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    extra_body = _build_reasoning_extra_body(settings)
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.3,
        extra_body=extra_body,
    )
