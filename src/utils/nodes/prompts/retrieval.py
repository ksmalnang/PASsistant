"""Retrieval node prompts."""

QUERY_REWRITE_PROMPT = """Given the user question below, extract a concise search query
that would best retrieve relevant academic policy documents. Focus on key terms,
regulations, and specific topics mentioned.

User question: {question}

Search query (Indonesian, concise, keyword-rich):"""
