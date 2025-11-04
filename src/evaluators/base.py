"""Base interface for evaluators."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from config.models import (
    EvaluationResult,
    MultiDimensionResult,
    EvaluationDimension,
)


class Evaluator(ABC):
    """Abstract base class for LLM-as-judge evaluators.
    
    Based on G-Eval methodology for evaluating text outputs.
    """
    
    def __init__(self, dimensions: Optional[List[EvaluationDimension]] = None):
        """Initialize evaluator with evaluation dimensions.
        
        Args:
            dimensions: List of evaluation dimensions (e.g., fluency, coherence)
        """
        self.dimensions = dimensions or []
    
    @abstractmethod
    def evaluate_dimension(
        self,
        dimension: EvaluationDimension,
        **context
    ) -> EvaluationResult:
        """Evaluate a single dimension.
        
        Args:
            dimension: The evaluation dimension
            **context: Context variables (e.g., Document, Summary, Question, Answer)
            
        Returns:
            EvaluationResult with aggregated score and raw samples
        """
        pass
    
    @abstractmethod
    def evaluate_multi_dimension(
        self,
        dimensions: List[EvaluationDimension],
        **context
    ) -> MultiDimensionResult:
        """Evaluate multiple dimensions.
        
        Args:
            dimensions: List of evaluation dimensions
            **context: Context variables
            
        Returns:
            MultiDimensionResult with scores for each dimension
        """
        pass
    
    @abstractmethod
    def batch_evaluate(
        self,
        dimension: EvaluationDimension,
        contexts: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Evaluate multiple items in batch for one dimension.
        
        Args:
            dimension: The evaluation dimension
            contexts: List of context dictionaries
            
        Returns:
            List of EvaluationResults
        """
        pass

