"""G-Eval evaluator implementation using GPT-4 or other LLMs."""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from evaluators.base import Evaluator
from evaluators.llm_client import LLMClient
from evaluators.scoring import parse_geval_score, aggregate_scores
from config.models import (
    EvaluationResult,
    MultiDimensionResult,
    EvaluationDimension,
)


class GEvalEvaluator(Evaluator):
    """G-Eval evaluator using LLMs with chain-of-thoughts and form-filling paradigm.
    
    Key features from G-Eval paper:
    1. High temperature (2.0) with multiple samples (n=20) for robustness
    2. Short max_tokens (5) to get scores only
    3. Score aggregation across samples
    4. Dimension-based evaluation (fluency, coherence, consistency, relevance)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        dimensions: Optional[List[EvaluationDimension]] = None,
        num_samples: int = 20,
        temperature: float = 2.0,
        max_tokens: int = 5,
        retry_delay: float = 0.5,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize G-Eval evaluator.
        
        Args:
            llm_client: LLM client with a generate() method. If not provided, 
                       creates one automatically using api_key/base_url/model
            dimensions: List of evaluation dimensions
            num_samples: Number of samples to generate per evaluation (n in paper)
            temperature: Temperature for sampling (paper uses 2.0)
            max_tokens: Max tokens for response (paper uses 5 for score only)
            retry_delay: Delay between retries in seconds
            max_retries: Maximum number of retries on error
            api_key: API key for LLM (only if llm_client not provided) 
            base_url: Base URL for LLM API (only if llm_client not provided)
            model: Model name (only if llm_client not provided)
        """
        super().__init__(dimensions)
        
        # Initialize LLM client if not provided
        if llm_client is None:
            self.llm_client = LLMClient(
                api_key=api_key,
                base_url=base_url,
                model=model
            )
        else:
            self.llm_client = llm_client
        
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_delay = retry_delay
        self.max_retries = max_retries
    
    def evaluate_dimension(
        self,
        dimension: EvaluationDimension,
        **context
    ) -> EvaluationResult:
        """Evaluate a single dimension using G-Eval methodology.
        
        Args:
            dimension: The evaluation dimension (e.g., fluency, coherence)
            **context: Context variables (e.g., Document="...", Summary="...")
            
        Returns:
            EvaluationResult with aggregated score and all raw scores
        """
        # Format prompt with context variables
        prompt = dimension.format_prompt(**context)
        
        # Generate multiple samples with high temperature
        all_responses = []
        raw_scores = []
        
        for attempt in range(self.max_retries):
            try:
                responses = self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.num_samples
                )
                
                # Parse scores from responses
                for response in responses:
                    score = parse_geval_score(
                        response,
                        min_score=dimension.min_score,
                        max_score=dimension.max_score
                    )
                    if score > 0:  # Valid score
                        raw_scores.append(score)
                        all_responses.append(response)
                
                break  # Success
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    # Return error result on final failure
                    return EvaluationResult(
                        dimension=dimension.name,
                        score=0.0,
                        raw_scores=[],
                        all_responses=[],
                        prompt=prompt,
                        metadata={"error": str(e)}
                    )
        
        # Aggregate scores
        if not raw_scores:
            aggregated_score = 0.0
        else:
            aggregated_score = aggregate_scores(raw_scores, method="mean")
        
        return EvaluationResult(
            dimension=dimension.name,
            score=aggregated_score,
            raw_scores=raw_scores,
            all_responses=all_responses,
            prompt=prompt,
            timestamp=datetime.now(),
            metadata={
                "num_samples": len(raw_scores),
                "expected_samples": self.num_samples,
                **context
            }
        )
    
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
        dimension_results = {}
        
        for dimension in dimensions:
            result = self.evaluate_dimension(dimension, **context)
            dimension_results[dimension.name] = result
        
        # Calculate overall score (simple average)
        scores = [r.score for r in dimension_results.values() if r.score > 0]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return MultiDimensionResult(
            dimensions=dimension_results,
            overall_score=overall_score,
            timestamp=datetime.now(),
            metadata=context
        )
    
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
        results = []
        
        for context in contexts:
            result = self.evaluate_dimension(dimension, **context)
            results.append(result)
            
            # Small delay to avoid rate limits
            time.sleep(self.retry_delay)
        
        return results
    
    def evaluate_with_human_scores(
        self,
        dimension: EvaluationDimension,
        contexts: List[Dict[str, Any]],
        human_scores: List[float]
    ) -> tuple[List[EvaluationResult], Dict[str, float]]:
        """Evaluate and compare with human scores for meta-evaluation.
        
        Args:
            dimension: The evaluation dimension
            contexts: List of context dictionaries
            human_scores: Corresponding human judgment scores
            
        Returns:
            Tuple of (evaluation_results, correlation_metrics)
        """
        from evaluators.scoring import calculate_correlations
        
        results = self.batch_evaluate(dimension, contexts)
        pred_scores = [r.score for r in results]
        
        # Calculate correlations
        correlations = calculate_correlations(pred_scores, human_scores)
        
        return results, correlations

