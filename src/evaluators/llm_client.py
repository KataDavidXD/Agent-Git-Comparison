"""LLM client for evaluators.

Note: This module re-exports LLMClient from the main package for convenient access.
The actual implementation is in llm_client
"""

from llm_client import LLMClient

__all__ = ["LLMClient"]

