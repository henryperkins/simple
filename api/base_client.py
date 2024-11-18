# base_client.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAIClient(ABC):
    """Base interface for all AI model clients."""
    
    @abstractmethod
    async def generate_docstring(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate docstring using the AI model."""
        pass
        
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of AI service."""
        pass

    @abstractmethod
    async def batch_process(self, prompts: List[str], **kwargs) -> List[Optional[Dict[str, Any]]]:
        """Process multiple prompts in batch."""
        pass
