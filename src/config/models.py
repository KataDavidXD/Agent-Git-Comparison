"""Configuration models for agent setup."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class ToolConfig:
    """Represents a tool configuration."""
    name: str
    description: str
    parameters: dict
    implementation: Optional[callable] = None


@dataclass
class PromptConfig:
    """Represents a prompt/instruction configuration."""
    role: str
    content: str
    template_vars: Optional[dict] = None


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    agent_id: str
    prompts: List[PromptConfig]
    tools: List[ToolConfig]
    metadata: Optional[dict] = None


@dataclass
class EvaluationDimension:
    """A single evaluation dimension for G-Eval style evaluation.
    
    Examples: fluency, coherence, consistency, relevance
    """
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5
    criteria: str = ""
    evaluation_steps: List[str] = field(default_factory=list)
    prompt_template: str = ""
    
    def format_prompt(self, **kwargs) -> str:
        """Format prompt template with variables (e.g., Document, Summary, Answer)."""
        prompt = self.prompt_template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            prompt = prompt.replace(placeholder, str(value))
        return prompt


@dataclass
class EvaluationResult:
    """Result of an evaluation with G-Eval methodology."""
    dimension: str  # Which dimension was evaluated
    score: float  # Aggregated score (e.g., mean of samples)
    raw_scores: List[float] = field(default_factory=list)  # All sample scores
    all_responses: List[str] = field(default_factory=list)  # LLM responses
    prompt: str = ""  # The prompt used
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def is_passing(self, threshold: float = 3.0) -> bool:
        """Check if result meets threshold (default for 1-5 scale)."""
        return self.score >= threshold
    
    @property
    def std_dev(self) -> float:
        """Standard deviation of raw scores."""
        if len(self.raw_scores) < 2:
            return 0.0
        mean = sum(self.raw_scores) / len(self.raw_scores)
        variance = sum((x - mean) ** 2 for x in self.raw_scores) / len(self.raw_scores)
        return variance ** 0.5


@dataclass
class MultiDimensionResult:
    """Result of evaluating multiple dimensions."""
    dimensions: Dict[str, EvaluationResult]  # dimension_name -> result
    overall_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    def get_dimension_score(self, dimension: str) -> Optional[float]:
        """Get score for a specific dimension."""
        result = self.dimensions.get(dimension)
        return result.score if result else None


@dataclass
class CorrelationResult:
    """Result of correlation analysis with human judgments."""
    pearson: float
    spearman: float
    kendall: float
    sample_size: int
    dimension: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
