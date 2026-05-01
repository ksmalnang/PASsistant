"""Prompt templates for node implementations."""

from src.utils.nodes.prompts.response import RESPONSE_SYSTEM_PROMPT
from src.utils.nodes.prompts.router import ROUTER_INTENT_PROMPT
from src.utils.nodes.prompts.student_record import STUDENT_RECORD_PROMPT

__all__ = [
    "RESPONSE_SYSTEM_PROMPT",
    "ROUTER_INTENT_PROMPT",
    "STUDENT_RECORD_PROMPT",
]
