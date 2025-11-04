"""LLM client for prompt generation.

Note: This module re-exports LLMClient from the main package for backward compatibility.
The actual implementation is in llm_client
"""

from llm_client import LLMClient
from llm_client import LLMClient_ChatOpenAI

__all__ = ["LLMClient", "LLMClient_ChatOpenAI"]

