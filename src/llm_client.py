"""LLM client for interacting with OpenAI-compatible APIs."""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import requests
import urllib3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class LLMClient_ChatOpenAI:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini",openai_api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("BASE_URL"), temperature=0.7)

    def generate(self, prompt: str):
        all_response = self.llm.invoke(prompt)
        content = all_response.content
        response = []
        response.append(content)
        return response

    def generate_with_system(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate with system and user prompts, return single string."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL"), 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        messages = [HumanMessage(content=f"System: {system_prompt}\n\nUser: {user_prompt}")]
        all_response = self.llm.invoke(messages)
        content = all_response.content
        return content  # Return string directly, not list

    def test_connection(self) -> bool:
        try:
            response = self.generate(prompt="Say 'Hello'")
            return len(response) > 0 and len(response[0]) > 0
        except Exception:
            return False


"""LLM client for interacting with OpenAI-compatible APIs."""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import requests


class LLMClient:
    """Client for connecting to OpenAI-compatible LLM APIs.
    
    Supports both prompt generation and G-Eval evaluation use cases.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize LLM client.
        
        Args:
            api_key: API key (reads from LLM_API_KEY env if not provided)
            base_url: Base URL for API (reads from LLM_BASE_URL env if not provided)
            model: Model name (reads from LLM_MODEL env if not provided)
        """
        # Load from .env file
        load_dotenv()
        
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        if not self.api_key:
            raise ValueError(
                "API key not found. Set LLM_API_KEY or OPENAI_API_KEY in .env file or pass as parameter."
            )
        
        # Ensure base_url ends without trailing slash
        self.base_url = self.base_url.rstrip('/')
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        n: int = 1,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """Generate text using LLM API.
        
        Args:
            prompt: The prompt text
            temperature: Temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate
            system_prompt: Optional system prompt
            
        Returns:
            List of generated texts (length = n)
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract all completions
            completions = []
            for choice in result["choices"]:
                content = choice["message"]["content"].strip()
                completions.append(content)
            
            return completions
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response format: {e}")
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate text with explicit system/user prompts.
        
        Convenience method for prompt generation use case.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt/query
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        results = self.generate(
            prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            system_prompt=system_prompt
        )
        return results[0]
    
    def test_connection(self) -> bool:
        """Test if the API connection is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate(
                prompt="Say 'Hello'",
                max_tokens=10,
                n=1
            )
            return len(response) > 0 and len(response[0]) > 0
        except Exception:
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get current client configuration.
        
        Returns:
            Dictionary with client configuration
        """
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_set": bool(self.api_key)
        }


if __name__ == "__main__":
    llm = LLMClient_ChatOpenAI()
    response = llm.generate("Hello, how are you?")
    print(response)

    """
    content="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?" additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 13, 'total_tokens': 44, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_efad92c60b', 'id': 'chatcmpl-CVYDewIxCVOpKR2R9Ni8S9aTAeguf', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--94344cdf-a924-406d-86a2-3fb9f622eb52-0' usage_metadata={'input_tokens': 13, 'output_tokens': 31, 'total_tokens': 44, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    """

    llm2 = LLMClient()
    response2 = llm2.generate("Hello, how are you?")
    print(response2)