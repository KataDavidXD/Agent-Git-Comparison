"""Base interfaces for prompt generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from config.models import PromptConfig


class PromptStyle(Enum):
    """Three prompt styles for A/B testing."""
    COT = "cot"  # Chain of Thought - step-by-step reasoning
    FEW_SHOT = "few_shot"  # Few-shot with examples
    CONCISE = "concise"  # Brief and to the point


@dataclass
class PromptVariation:
    """A variation of a prompt with specific style."""
    style: PromptStyle
    content: str
    config: PromptConfig
    metadata: Optional[Dict[str, Any]] = None


class PromptGenerator(ABC):
    """Abstract base class for prompt generators.
    
    Generates three styled variations from a draft prompt:
    1. CoT (Chain of Thought): Encourages step-by-step reasoning
    2. Few-shot: Includes examples to guide the model
    3. Concise: Brief and direct instruction
    """
    
    @abstractmethod
    def generate_styled_prompts(
        self,
        draft_prompt: str,
        role: str = "system",
        examples: Optional[List[Dict[str, str]]] = None
    ) -> List[PromptVariation]:
        """Generate three styled variations from a draft prompt.
        
        Args:
            draft_prompt: The base prompt to transform
            role: Role for the prompt (default: "system")
            examples: Optional examples for few-shot style
            
        Returns:
            List of three PromptVariation objects (CoT, Few-shot, Concise)
        """
        pass
    
    @abstractmethod
    def generate_cot_prompt(self, draft_prompt: str) -> str:
        """Generate Chain of Thought version of the prompt."""
        pass
    
    @abstractmethod
    def generate_few_shot_prompt(
        self,
        draft_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate Few-shot version with examples."""
        pass
    
    @abstractmethod
    def generate_concise_prompt(self, draft_prompt: str) -> str:
        """Generate concise version of the prompt."""
        pass

