"""Evaluation module for LLM-as-judge based on G-Eval."""

from .base import Evaluator
from .geval_evaluator import GEvalEvaluator
from .llm_client import LLMClient
from .scoring import (
    parse_geval_score,
    aggregate_scores,
    calculate_correlations,
    calculate_score_statistics,
)
from . import prompt_templates

__all__ = [
    "Evaluator",
    "GEvalEvaluator",
    "LLMClient",
    "parse_geval_score",
    "aggregate_scores",
    "calculate_correlations",
    "calculate_score_statistics",
    "prompt_templates",
]

