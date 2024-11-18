# model_factory.py
from typing import Optional, Dict
from ..base_client import BaseAIClient
from core.config import AIModelConfig
from api.models.api_client import AzureOpenAIClient
from api.models.openai_model import OpenAIClient
from api.models.claude_model import ClaudeClient
from api.models.gemini_model import GeminiClient

class AIClientFactory:
    """Factory for creating AI model clients with unified interface."""
    
    _clients: Dict[str, type] = {
        "azure": AzureOpenAIClient,
        "openai": OpenAIClient,
        "claude": ClaudeClient,
        "gemini": GeminiClient
    }
    
    @classmethod
    def register_client(cls, model_type: str, client_class: type):
        """Register new model client."""
        cls._clients[model_type] = client_class
    
    @classmethod
    def create_client(cls, config: AIModelConfig) -> Optional[BaseAIClient]:
        """Create appropriate AI client based on configuration."""
        if config.model_type not in cls._clients:
            raise ValueError(f"Unsupported model type: {config.model_type}")
            
        client_class = cls._clients[config.model_type]
        return client_class(config)
