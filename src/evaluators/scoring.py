"""Scoring utilities for G-Eval."""

from typing import List, Dict, Any, Optional
import re


def parse_geval_score(response: str, min_score: int = 1, max_score: int = 5) -> float:
    """Parse score from G-Eval LLM response.
    
    G-Eval uses short responses (max_tokens=5) to get scores directly.
    Expected formats: "3", "3.5", " 4", etc.
    
    Args:
        response: The LLM response text
        min_score: Minimum valid score
        max_score: Maximum valid score
        
    Returns:
        Parsed score, or 0 if parsing fails
    """
    try:
        # Clean the response
        text = response.strip()
        
        # Try to find first number
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            score = float(match.group(1))
            # Validate range
            if min_score <= score <= max_score:
                return score
        
        return 0.0
        
    except Exception:
        return 0.0


def aggregate_scores(scores: List[float], method: str = "mean") -> float:
    """Aggregate multiple scores using specified method.
    
    Args:
        scores: List of scores to aggregate
        method: Aggregation method ("mean", "median", "max", "min")
        
    Returns:
        Aggregated score
    """
    if not scores:
        return 0.0
    
    if method == "mean":
        return sum(scores) / len(scores)
    elif method == "median":
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
        else:
            return sorted_scores[mid]
    elif method == "max":
        return max(scores)
    elif method == "min":
        return min(scores)
    else:
        return sum(scores) / len(scores)  # Default to mean


def calculate_correlations(
    pred_scores: List[float],
    human_scores: List[float]
) -> Dict[str, float]:
    """Calculate correlation metrics between predicted and human scores.
    
    Based on G-Eval meta-evaluation methodology.
    
    Args:
        pred_scores: Predicted scores from evaluator
        human_scores: Human judgment scores
        
    Returns:
        Dictionary with pearson, spearman, and kendall correlations
    """
    try:
        from scipy.stats import pearsonr, spearmanr, kendalltau
        
        if len(pred_scores) != len(human_scores):
            raise ValueError("Score lists must have same length")
        
        if len(pred_scores) < 2:
            return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
        
        # Filter out zero scores (failed evaluations)
        valid_pairs = [
            (p, h) for p, h in zip(pred_scores, human_scores)
            if p > 0
        ]
        
        if len(valid_pairs) < 2:
            return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
        
        valid_pred = [p for p, h in valid_pairs]
        valid_human = [h for p, h in valid_pairs]
        
        # Check for variance
        if len(set(valid_pred)) <= 1 or len(set(valid_human)) <= 1:
            return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}
        
        pearson = pearsonr(valid_pred, valid_human)[0]
        spearman = spearmanr(valid_pred, valid_human)[0]
        kendall = kendalltau(valid_pred, valid_human)[0]
        
        return {
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall,
            "sample_size": len(valid_pairs)
        }
        
    except ImportError:
        # scipy not available
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "kendall": 0.0,
            "error": "scipy not installed"
        }
    except Exception as e:
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "kendall": 0.0,
            "error": str(e)
        }


def calculate_score_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate descriptive statistics for a list of scores.
    
    Args:
        scores: List of scores
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }
    
    mean = sum(scores) / len(scores)
    
    # Standard deviation
    if len(scores) > 1:
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        std = variance ** 0.5
    else:
        std = 0.0
    
    # Median
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
    else:
        median = sorted_scores[mid]
    
    return {
        "count": len(scores),
        "mean": mean,
        "std": std,
        "min": min(scores),
        "max": max(scores),
        "median": median
    }

