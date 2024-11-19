"""
AI Interaction Handler with proper integration of token management,
response parsing, and monitoring.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from openai import AsyncAzureOpenAI, OpenAIError
from core.logger import log_info, log_error, log_debug
from core.cache import Cache
from core.config import AzureOpenAIConfig
from api.token_management import TokenManager, TokenUsage
from api.response_parser import ResponseParser
from core.monitoring import SystemMonitor, ModelMetrics, APIMetrics
from docs.docstring_utils import DocstringValidator

class AIInteractionHandler:
    """Streamlined handler for AI model interactions with proper integrations."""

    def __init__(
        self,
        config: AzureOpenAIConfig,
        cache_config: Optional[Dict] = None,
        batch_size: int = 5
    ):
        """Initialize the interaction handler with all necessary components."""
        self.config = config
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        
        # Initialize all required components
        self.token_manager = TokenManager(
            model=config.model_name,
            deployment_name=config.deployment_name
        )
        self.response_parser = ResponseParser()
        self.monitor = SystemMonitor()
        self.cache = Cache(**(cache_config or {}))
        self.validator = DocstringValidator()
        self.batch_size = batch_size
        
        log_info("AI Interaction Handler initialized with all components")

    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
        is_class: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Generate a docstring using the AI model with full monitoring and token management."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Check cache first
            cache_key = f"docstring:{func_name}:{hash(str(params))}"
            cached = await self.cache.get_cached_docstring(cache_key)
            if cached:
                self.monitor.log_cache_hit(func_name)
                return cached

            self.monitor.log_cache_miss(func_name)

            # Create messages
            messages = self._create_messages(
                func_name, params, return_type, complexity_score,
                existing_docstring, decorators, exceptions, is_class
            )

            # Token validation and optimization
            prompt_text = json.dumps(messages)
            prompt_tokens = self.token_manager.estimate_tokens(prompt_text)
            
            # Validate token limits
            is_valid, metrics, message = self.token_manager.validate_request(prompt_text)
            if not is_valid:
                log_error(f"Token validation failed for {func_name}: {message}")
                self.monitor.log_operation_complete(func_name, 0, 0, error=message)
                return None

            # Optimize if needed
            if prompt_tokens > self.config.max_tokens * 0.8:
                messages, token_usage = self.token_manager.optimize_prompt(
                    prompt_text,
                    max_tokens=self.config.max_tokens,
                    preserve_sections=['parameters', 'returns']
                )
                log_info(f"Optimized prompt from {prompt_tokens} to {token_usage.prompt_tokens} tokens")

            # Make API request with monitoring
            response = await self._make_api_request(messages, func_name)
            if not response:
                return None

            # Parse and validate response
            parsed_response = await self._process_response(response, func_name)
            if not parsed_response:
                return None

            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time

            # Log metrics
            self._log_operation_metrics(
                func_name=func_name,
                start_time=start_time,
                response_time=response_time,
                response=response,
                parsed_response=parsed_response
            )

            # Cache valid response
            if parsed_response:
                await self.cache.save_docstring(cache_key, parsed_response)

            return parsed_response

        except Exception as e:
            log_error(f"Error generating docstring for {func_name}: {str(e)}")
            self.monitor.log_operation_complete(
                func_name,
                asyncio.get_event_loop().time() - start_time,
                0,
                error=str(e)
            )
            return None

    async def _make_api_request(self, messages: List[Dict[str, str]], context: str) -> Optional[Any]:
        """Make an API request with full monitoring and token management."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Track token usage
            if response.usage:
                self.token_manager.track_request(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )

            # Log API metrics
            self.monitor.log_api_request(APIMetrics(
                endpoint=self.config.endpoint,
                tokens=response.usage.total_tokens if response.usage else 0,
                response_time=asyncio.get_event_loop().time() - start_time,
                status="success",
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                estimated_cost=self.token_manager.estimate_cost(
                    response.usage.prompt_tokens if response.usage else 0,
                    response.usage.completion_tokens if response.usage else 0
                )
            ))

            return response

        except Exception as e:
            log_error(f"API request failed: {str(e)}")
            self.monitor.log_api_request(APIMetrics(
                endpoint=self.config.endpoint,
                tokens=0,
                response_time=asyncio.get_event_loop().time() - start_time,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                estimated_cost=0,
                error=str(e)
            ))
            return None

    async def _process_response(
        self,
        response: Any,
        context: str
    ) -> Optional[Dict[str, Any]]:
        """Process response with response parser and validation."""
        try:
            # Use response parser
            parsed_response = self.response_parser.parse_json_response(
                response.choices[0].message.content
            )
            
            if not parsed_response:
                log_error(f"Failed to parse response for {context}")
                return None

            # Validate response format
            if not self.response_parser.validate_response(parsed_response):
                log_error(f"Invalid response format for {context}")
                return None

            # Validate docstring content
            is_valid, validation_errors = self.validator.validate_docstring(
                parsed_response
            )
            
            if not is_valid:
                log_error(f"Docstring validation failed for {context}: {validation_errors}")
                return None

            return parsed_response

        except Exception as e:
            log_error(f"Response processing error: {str(e)}")
            return None

    def _log_operation_metrics(
        self,
        func_name: str,
        start_time: float,
        response_time: float,
        response: Any,
        parsed_response: Optional[Dict[str, Any]]
    ) -> None:
        """Log comprehensive operation metrics."""
        # Log model metrics
        self.monitor.log_model_metrics("azure", ModelMetrics(
            model_type="azure",
            operation=func_name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            response_time=response_time,
            status="success" if parsed_response else "error",
            cost=self.token_manager.estimate_cost(
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0
            )
        ))

        # Log operation completion
        self.monitor.log_operation_complete(
            func_name,
            response_time,
            response.usage.total_tokens if response.usage else 0
        )

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "monitor_metrics": self.monitor.get_metrics_summary(),
            "token_usage": self.token_manager.get_usage_stats(),
            "cache_stats": await self.cache.get_stats()
        }

    async def close(self):
        """Close all components properly."""
        try:
            await self.client.close()
            await self.cache.close()
            log_info("AI Interaction Handler closed")
        except Exception as e:
            log_error(f"Error closing handler: {str(e)}")

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()