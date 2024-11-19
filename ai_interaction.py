# ai_interaction.py
"""
AI Interaction Handler Module

Manages interactions with AI models, including token management, caching,
response parsing, and monitoring.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from openai import AsyncAzureOpenAI

from core.logger import log_info, log_error, log_debug
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.monitoring import MetricsCollector, SystemMonitor
from core.exceptions import (
   AIServiceError,
   TokenLimitError,
   ValidationError,
   ProcessingError
)
from core.token_management import TokenManager
from core.response_parser import ResponseParser
from core.extraction_manager import ExtractionManager
from core.docs import DocStringManager

class AIInteractionHandler:
   """Manages AI model interactions with integrated monitoring and caching."""

   def __init__(
       self,
       config: AzureOpenAIConfig,
       cache: Optional[Cache] = None,
       batch_size: int = 5,
       metrics_collector: Optional[MetricsCollector] = None
   ):
       """
       Initialize the AI interaction handler.

       Args:
           config: Configuration for Azure OpenAI
           cache: Optional cache instance
           batch_size: Size of batches for processing
           metrics_collector: Optional metrics collector instance
       """
       self.config = config
       self.cache = cache
       self.batch_size = batch_size
       
       # Initialize components
       self.client = AsyncAzureOpenAI(
           api_key=config.api_key,
           api_version=config.api_version,
           azure_endpoint=config.endpoint
       )
       
       self.token_manager = TokenManager(
           model=config.model_name,
           deployment_name=config.deployment_name
       )
       
       self.response_parser = ResponseParser(self.token_manager)
       self.monitor = SystemMonitor()
       self.metrics = metrics_collector or MetricsCollector()
       
       log_info("AI Interaction Handler initialized")

   async def process_code(self, source_code: str) -> Tuple[str, str]:
       """
       Process source code to generate documentation.
       
       Args:
           source_code: Source code to process
           
       Returns:
           Tuple[str, str]: (updated_code, documentation)
           
       Raises:
           ProcessingError: If processing fails
       """
       try:
           operation_start = datetime.now()
           
           # Extract metadata
           extractor = ExtractionManager()
           metadata = extractor.extract_metadata(source_code)
           
           # Process functions and classes in batches
           doc_entries = []
           
           # Process functions
           for batch in self._batch_items(metadata['functions'], self.batch_size):
               batch_results = await asyncio.gather(*[
                   self.generate_docstring(
                       func_name=func['name'],
                       params=func['args'],
                       return_type=func['return_type'],
                       complexity_score=func.get('complexity', 0),
                       existing_docstring=func.get('docstring', ''),
                       decorators=func.get('decorators', []),
                       exceptions=func.get('exceptions', [])
                   )
                   for func in batch
               ], return_exceptions=True)
               
               for func, result in zip(batch, batch_results):
                   if isinstance(result, Exception):
                       log_error(f"Error processing function {func['name']}: {str(result)}")
                       continue
                   if result:
                       doc_entries.append({
                           'type': 'function',
                           'name': func['name'],
                           'docstring': result['docstring']
                       })

           # Process classes
           for batch in self._batch_items(metadata['classes'], self.batch_size):
               batch_results = await asyncio.gather(*[
                   self.generate_docstring(
                       func_name=cls['name'],
                       params=[],
                       return_type='None',
                       complexity_score=cls.get('complexity', 0),
                       existing_docstring=cls.get('docstring', ''),
                       decorators=cls.get('decorators', []),
                       is_class=True
                   )
                   for cls in batch
               ], return_exceptions=True)
               
               for cls, result in zip(batch, batch_results):
                   if isinstance(result, Exception):
                       log_error(f"Error processing class {cls['name']}: {str(result)}")
                       continue
                   if result:
                       doc_entries.append({
                           'type': 'class',
                           'name': cls['name'],
                           'docstring': result['docstring']
                       })

           # Process documentation
           doc_manager = DocStringManager(source_code)
           result = doc_manager.process_batch(doc_entries)
           
           # Track metrics
           operation_time = (datetime.now() - operation_start).total_seconds()
           self.metrics.track_operation(
               operation_type='process_code',
               success=bool(result),
               duration=operation_time
           )
           
           if result:
               return result['code'], result['documentation']
           raise ProcessingError("Failed to generate documentation")
           
       except Exception as e:
           log_error(f"Error processing code: {str(e)}")
           self.metrics.track_operation(
               operation_type='process_code',
               success=False,
               error=str(e)
           )
           raise ProcessingError(f"Code processing failed: {str(e)}")

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
       """
       Generate a docstring using the AI model.
       
       Args:
           func_name: Name of the function/class
           params: List of parameter tuples (name, type)
           return_type: Return type annotation
           complexity_score: Code complexity score
           existing_docstring: Existing docstring if any
           decorators: List of decorators
           exceptions: List of exceptions
           is_class: Whether generating for a class
           
       Returns:
           Optional[Dict[str, Any]]: Generated docstring data if successful
       """
       operation_start = datetime.now()
       
       try:
           # Check cache first
           if self.cache:
               cache_key = self._generate_cache_key(
                   func_name, params, return_type, complexity_score, is_class
               )
               cached = await self.cache.get_cached_docstring(cache_key)
               if cached:
                   self.metrics.track_cache_hit()
                   return cached
               self.metrics.track_cache_miss()

           # Create messages for AI model
           messages = self._create_messages(
               func_name, params, return_type, complexity_score,
               existing_docstring, decorators, exceptions, is_class
           )

           # Validate token limits
           prompt_text = json.dumps(messages)
           is_valid, token_metrics = self.token_manager.validate_request(prompt_text)
           if not is_valid:
               raise TokenLimitError(f"Token validation failed: {token_metrics}")

           # Make API request
           response = await self._make_api_request(messages, func_name)
           if not response:
               raise AIServiceError("Empty response from AI service")

           # Parse and validate response
           parsed_response = await self._process_response(response, func_name)
           if not parsed_response:
               raise ValidationError("Failed to parse AI response")

           # Cache valid response
           if self.cache and parsed_response:
               await self.cache.save_docstring(cache_key, parsed_response)

           # Track metrics
           operation_time = (datetime.now() - operation_start).total_seconds()
           self.metrics.track_operation(
               operation_type='generate_docstring',
               success=True,
               duration=operation_time
           )

           return parsed_response

       except Exception as e:
           operation_time = (datetime.now() - operation_start).total_seconds()
           self.metrics.track_operation(
               operation_type='generate_docstring',
               success=False,
               duration=operation_time,
               error=str(e)
           )
           log_error(f"Error generating docstring for {func_name}: {str(e)}")
           raise

   def _generate_cache_key(
       self,
       func_name: str,
       params: List[Tuple[str, str]],
       return_type: str,
       complexity_score: int,
       is_class: bool
   ) -> str:
       """Generate a consistent cache key."""
       key_parts = [
           func_name,
           str(sorted(params)),
           return_type,
           str(complexity_score),
           str(is_class)
       ]
       return f"docstring:{':'.join(key_parts)}"

   def _create_messages(
       self,
       func_name: str,
       params: List[Tuple[str, str]],
       return_type: str,
       complexity_score: int,
       existing_docstring: str,
       decorators: Optional[List[str]],
       exceptions: Optional[List[str]],
       is_class: bool
   ) -> List[Dict[str, str]]:
       """Create messages for AI model prompt."""
       return [
           {
               "role": "system",
               "content": "Generate clear, comprehensive docstrings following Google style guide."
           },
           {
               "role": "user",
               "content": json.dumps({
                   "name": func_name,
                   "type": "class" if is_class else "function",
                   "parameters": [{"name": p[0], "type": p[1]} for p in params],
                   "return_type": return_type,
                   "complexity_score": complexity_score,
                   "existing_docstring": existing_docstring,
                   "decorators": decorators or [],
                   "exceptions": exceptions or []
               })
           }
       ]

   async def _make_api_request(
       self,
       messages: List[Dict[str, str]],
       context: str
   ) -> Optional[Dict[str, Any]]:
       """Make an API request with monitoring and token management."""
       operation_start = datetime.now()
       
       try:
           response = await self.client.chat.completions.create(
               model=self.config.deployment_name,
               messages=messages,
               temperature=self.config.temperature,
               max_tokens=self.config.max_tokens
           )

           if response and response.choices:
               # Track token usage
               usage = response.usage
               if usage:
                   self.token_manager.track_request(
                       usage.prompt_tokens,
                       usage.completion_tokens
                   )

               operation_time = (datetime.now() - operation_start).total_seconds()
               self.metrics.track_operation(
                   operation_type='api_request',
                   success=True,
                   duration=operation_time,
                   tokens_used=usage.total_tokens if usage else 0
               )

               return {
                   "content": response.choices[0].message.content,
                   "usage": {
                       "prompt_tokens": usage.prompt_tokens if usage else 0,
                       "completion_tokens": usage.completion_tokens if usage else 0,
                       "total_tokens": usage.total_tokens if usage else 0
                   }
               }

           return None

       except Exception as e:
           operation_time = (datetime.now() - operation_start).total_seconds()
           self.metrics.track_operation(
               operation_type='api_request',
               success=False,
               duration=operation_time,
               error=str(e)
           )
           log_error(f"API request failed for {context}: {str(e)}")
           raise AIServiceError(f"API request failed: {str(e)}")

   async def _process_response(
       self,
       response: Dict[str, Any],
       context: str
   ) -> Optional[Dict[str, Any]]:
       """Process and validate AI response."""
       try:
           if not response.get("content"):
               return None

           parsed_response = self.response_parser.parse_json_response(
               response["content"]
           )
           
           if not parsed_response:
               return None

           if not self.response_parser.validate_response(parsed_response):
               return None

           return parsed_response

       except Exception as e:
           log_error(f"Response processing error for {context}: {str(e)}")
           return None

   @staticmethod
   def _batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
       """Split items into batches."""
       return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

   async def get_metrics_summary(self) -> Dict[str, Any]:
       """Get comprehensive metrics summary."""
       try:
           cache_stats = await self.cache.get_stats() if self.cache else {}
           
           return {
               "metrics": self.metrics.get_metrics(),
               "token_usage": self.token_manager.get_usage_stats(),
               "cache_stats": cache_stats,
               "monitor_metrics": self.monitor.get_metrics_summary()
           }
       except Exception as e:
           log_error(f"Error getting metrics summary: {str(e)}")
           return {}

   async def close(self) -> None:
       """Close all components properly."""
       try:
           if self.cache:
               await self.cache.close()
           self.token_manager.reset_cache()
           self.monitor.reset()
           log_info("AI Interaction Handler closed successfully")
       except Exception as e:
           log_error(f"Error closing handler: {str(e)}")

   async def __aenter__(self):
       """Async context manager entry."""
       return self

   async def __aexit__(self, exc_type, exc_val, exc_tb):
       """Async context manager exit."""
       await self.close()