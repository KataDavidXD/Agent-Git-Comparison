"""Concrete implementation of prompt generator using LLM."""

from typing import List, Dict, Optional

from generators.base import (
    PromptGenerator,
    PromptStyle,
    PromptVariation,
)
#from generators.llm_client import LLMClient
from llm_client import LLMClient_ChatOpenAI
from config.models import PromptConfig


class SimplePromptGenerator(PromptGenerator):
    """LLM-based prompt generator.
    
    Generates three styled variations from a draft prompt using an LLM:
    1. CoT: Chain of Thought - step-by-step reasoning
    2. Few-shot: Includes examples to guide the model
    3. Concise: Brief and direct instruction
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize the generator with LLM client.
        
        Args:
            api_key: LLM API key (reads from .env if not provided)
            base_url: LLM API base URL (reads from .env if not provided)
            model: LLM model name (reads from .env if not provided)
        """
        # self.llm_client = LLMClient(
        #     api_key=api_key,
        #     base_url=base_url,
        #     model=model
        # )
        self.llm_client = LLMClient_ChatOpenAI()
    
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
                     Format: [{"input": "...", "output": "..."}, ...]
            
        Returns:
            List of three PromptVariation objects (CoT, Few-shot, Concise)
        """
        variations = []
        
        # Generate CoT version
        cot_content = self.generate_cot_prompt(draft_prompt)
        cot_config = PromptConfig(
            role=role,
            content=cot_content,
            template_vars={"style": "cot", "source": "generated"}
        )
        variations.append(PromptVariation(
            style=PromptStyle.COT,
            content=cot_content,
            config=cot_config,
            metadata={"description": "Chain of Thought - step-by-step reasoning"}
        ))
        
        # Generate Few-shot version
        few_shot_content = self.generate_few_shot_prompt(draft_prompt, examples)
        few_shot_config = PromptConfig(
            role=role,
            content=few_shot_content,
            template_vars={"style": "few_shot", "source": "generated"}
        )
        variations.append(PromptVariation(
            style=PromptStyle.FEW_SHOT,
            content=few_shot_content,
            config=few_shot_config,
            metadata={"description": "Few-shot learning with examples"}
        ))
        
        # Generate Concise version
        concise_content = self.generate_concise_prompt(draft_prompt)
        concise_config = PromptConfig(
            role=role,
            content=concise_content,
            template_vars={"style": "concise", "source": "generated"}
        )
        variations.append(PromptVariation(
            style=PromptStyle.CONCISE,
            content=concise_content,
            config=concise_config,
            metadata={"description": "Concise and direct instruction"}
        ))
        
        return variations
    
    def generate_cot_prompt(self, draft_prompt: str) -> str:
        """Generate Chain of Thought version with step-by-step reasoning.
        
        Creates a system message that encourages CoT thinking.
        Output will be wrapped in message format by the loader.
        """
        system_prompt = """You are an expert prompt engineer specializing in Chain of Thought (CoT) prompting.

Transform the draft into a CoT-optimized system message that:
1. Explicitly instructs step-by-step reasoning
2. Includes phrases like "Let's think step by step" or "Break this down:"
3. Encourages showing work and intermediate steps
4. Structures the reasoning process

Output ONLY the system message content - no JSON, no role labels."""

        user_prompt = f"""Transform this draft into a CoT system message:

{draft_prompt.strip()}

System message:"""

        return self.llm_client.generate_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7
        )
    
    def generate_few_shot_prompt(
        self,
        draft_prompt: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate Few-shot version with demonstration examples.
        
        Args:
            draft_prompt: Base prompt
            examples: List of input/output example pairs. If not provided, LLM will create them.
            
        Creates a system message with integrated examples.
        Output will be wrapped in message format by the loader.
        """
        system_prompt = """You are an expert prompt engineer specializing in Few-shot learning prompts.

Transform the draft into a Few-shot system message that:
1. Clearly describes the task at the beginning
2. Provides 2-3 high-quality example demonstrations
3. Uses consistent format for each example: "Input: ... Output: ..."
4. Examples must be realistic and directly relevant to the task
5. Ends with clear instruction to follow the demonstrated pattern

IMPORTANT: The examples MUST be included in the system message itself as demonstrations.

Output ONLY the system message content - no JSON, no role labels."""

        examples_text = ""
        if examples:
            examples_text = "\n\nUse these provided examples in the Few-shot prompt:\n"
            for i, ex in enumerate(examples[:3], 1):
                examples_text += f"Example {i} - Input: {ex.get('input', 'N/A')} | Output: {ex.get('output', 'N/A')}\n"
            examples_text += "\nIncorporate these examples into the system message as demonstrations."
        else:
            examples_text = "\n\nNo examples provided. You MUST create 2-3 realistic, task-specific examples and include them as demonstrations in the system message. Make sure the examples clearly show the expected input-output pattern for this task."

        user_prompt = f"""Transform this draft into a Few-shot system message with demonstrations:

Draft task: {draft_prompt.strip()}{examples_text}

Generate a complete Few-shot system message with examples integrated:"""

        return self.llm_client.generate_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7
        )
    
    def generate_concise_prompt(self, draft_prompt: str) -> str:
        """Generate concise, direct version without unnecessary words.
        
        Creates a brief system message that gets straight to the point.
        Output will be wrapped in message format by the loader.
        """
        system_prompt = """You are an expert prompt engineer specializing in concise communication.

Transform the draft into a Concise system message that:
1. Removes all filler words and unnecessary politeness
2. Uses direct imperative verbs
3. Gets straight to the core task
4. Keeps only essential information
5. Makes instructions clear and actionable

Output ONLY the system message content - no JSON, no role labels."""

        user_prompt = f"""Transform this draft into a Concise system message:

{draft_prompt.strip()}

System message:"""

        return self.llm_client.generate_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7
        )

