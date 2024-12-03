"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API for docstring generation, handling token management,
caching, and response processing with structured JSON outputs.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union, List
import ast
from dataclasses import dataclass
import openai
from openai import APIError
from jsonschema import validate, ValidationError as JsonValidationError
from core.logger import LoggerSetup
from core.cache import Cache
from core.metrics import Metrics
from core.metrics_collector import MetricsCollector
from core.config import AzureOpenAIConfig
from core.extraction.code_extractor import CodeExtractor
from core.extraction.types import ExtractionContext
from core.docstring_processor import DocstringProcessor
from core.types import ProcessingResult, DocstringData, AIHandler, DocumentationContext
from api.token_management import TokenManager
from api.api_client import APIClient
from core.schema_loader import load_schema
from exceptions import ValidationError, ExtractionError

logger = LoggerSetup.get_logger(__name__)


DOCSTRING_SCHEMA = load_schema('docstring_schema')

class AIInteractionHandler(AIHandler):
    """
    Handles AI interactions for docstring generation via Azure OpenAI API.

    Attributes:
        cache (Optional[Cache]): Optional cache for storing intermediate results.
        metrics_collector (Optional[MetricsCollector]): Optional metrics collector for tracking operations.
        token_manager (Optional[TokenManager]): Optional token manager for handling token limits.
        config (Optional[AzureOpenAIConfig]): Configuration settings for Azure OpenAI service.
        client (APIClient): Client to interact with Azure OpenAI API.
        docstring_processor (DocstringProcessor): Processor for handling docstring operations.
        code_extractor (CodeExtractor): Extractor for analyzing and extracting code elements.
        context (ExtractionContext): Context for extraction operations.
    """
    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_calculator: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None,
        config: Optional[AzureOpenAIConfig] = None,
        code_extractor: Optional[CodeExtractor] = None
    ) -> None:
        """Initialize the AI Interaction Handler."""
        self.logger = LoggerSetup.get_logger(__name__)

        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.metrics_calculator = metrics_calculator or MetricsCollector()
            self.token_manager = token_manager or TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config,
                metrics_collector=self.metrics_calculator
            )
            self.client = APIClient(self.config)
            self.docstring_processor = DocstringProcessor()
            self.code_extractor = code_extractor or CodeExtractor()
            self.context = ExtractionContext()
            self._initialize_tools()
            self.logger.info("AI Interaction Handler initialized successfully")

            openai.api_type = "azure"
            openai.api_key = self.config.api_key
            openai.api_base_url = self.config.endpoint
            openai.api_version = self.config.api_version

            self._current_module_tree = None
            self._current_module_docs = {}
            self._current_module = None

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code before parsing to handle special cases."""
        try:
            import re
            timestamp_pattern = r'($|\b)(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)$?'
            
            def timestamp_replacer(match):
                timestamp = match.group(2)
                prefix = match.group(1) if match.group(1) == '[' else ''
                suffix = ']' if prefix == '[' else ''
                return f'{prefix}"{timestamp}"{suffix}'

            processed_code = re.sub(timestamp_pattern, timestamp_replacer, source_code)
            self.logger.debug("Preprocessed source code to handle timestamps")
            return processed_code

        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    def _create_method_info(self, method) -> Dict[str, Any]:
        """Create method information dictionary."""
        return {
            'name': method.name,
            'docstring': method.docstring,
            'args': [{'name': arg.name, 'type': arg.type_hint} for arg in method.args],
            'return_type': method.return_type,
            'complexity': method.metrics.get('complexity', 0) if method.metrics else 0
        }

    def _create_class_info(self, cls) -> Dict[str, Any]:
        """Create class information dictionary."""
        return {
            'name': cls.name,
            'docstring': cls.docstring,
            'methods': [self._create_method_info(method) for method in cls.methods],
            'complexity': cls.metrics.get('complexity', 0) if cls.metrics else 0
        }

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        context: Optional[ExtractionContext] = None
    ) -> Optional[Tuple[str, str]]:
        """Process source code to generate documentation."""
        try:
            # Check cache first
            if cache_key and self.cache:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        updated_code = cached_result.get("updated_code")
                        documentation = cached_result.get("documentation")
                        if isinstance(updated_code, str) and isinstance(documentation, str):
                            return updated_code, documentation
                        return None
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Preprocess code
            processed_code = self._preprocess_code(source_code)
            
            # Parse AST once
            try:
                tree = ast.parse(processed_code)
            except SyntaxError as e:
                self.logger.error(f"Syntax error in source code: {e}")
                raise ExtractionError(f"Failed to parse code: {e}")

            # Extract metadata
            if extracted_info:
                metadata = extracted_info
            else:
                self.logger.debug("Processing code with extractor...")
                active_context = context if context is not None else self.context
                extraction_result = self.code_extractor.extract_code(processed_code, active_context)

                if not extraction_result:
                    self.logger.error("Failed to process code with extractor")
                    return None

                metadata = {
                    'module_docstring': extraction_result.module_docstring,
                    'classes': [self._create_class_info(cls) for cls in extraction_result.classes],
                    'functions': [self._create_method_info(func) for func in extraction_result.functions],
                    'metrics': extraction_result.metrics or {}
                }
            
            # Generate prompt and get AI response
            prompt = self._create_function_calling_prompt(processed_code, metadata)
            
            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
                tools=[self.docstring_function],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                self.logger.error("No response from AI service")
                return None

            # Extract tool calls from response
            message = response['choices'][0]['message']
            if not message.get('tool_calls') or not message['tool_calls'][0].get('function'):
                self.logger.error("No function call in response")
                return None

            # Parse function arguments
            function_args = message['tool_calls'][0]['function'].get('arguments', '{}')
            try:
                response_data = json.loads(function_args)
                self.logger.debug(f"Parsed response data: {json.dumps(response_data, indent=2)}")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error - Raw content that failed: {function_args}")
                response_data = self._create_fallback_response({})

            # Validate response data
            try:
                validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
            except ValidationError as e:
                self.logger.error(f"Validation error - Schema path: {' -> '.join(str(p) for p in e.path)}")
                self.logger.error(f"Validation error message: {e.message}")
                response_data = self._create_fallback_response(response_data)

            # Create docstring data
            try:
                docstring_data = DocstringData(**response_data)
                self.logger.debug(f"Created DocstringData object successfully: {docstring_data}")
            except Exception as e:
                self.logger.error(f"Error creating DocstringData: {e}")
                return None

            # Format docstring
            docstring = self.docstring_processor.format(docstring_data)
            self.logger.debug(f"Formatted docstring result: {docstring}")

            # Update AST with new docstrings
            try:
                modified = False
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        existing_docstring = ast.get_docstring(node) or ""
                        updated_docstring = self.docstring_processor.update_docstring(
                            existing_docstring, docstring
                        )
                        self.docstring_processor.insert_docstring(node, updated_docstring)
                        modified = True

                # Convert updated AST back to source code
                updated_code = ast.unparse(tree) if modified else processed_code
            except Exception as e:
                self.logger.error(f"Error updating source code: {e}")
                return None

            # Track token usage
            if usage:
                self.token_manager.track_request(
                    usage.get('prompt_tokens', 0),
                    usage.get('completion_tokens', 0)
                )

            # Cache results
            if cache_key and self.cache:
                try:
                    await self.cache.save_docstring(
                        cache_key,
                        {
                            "updated_code": updated_code,
                            "documentation": docstring
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Cache save error: {e}")

            return updated_code, docstring

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    def _create_fallback_response(self, partial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a valid response from partial/invalid data."""
        fallback = {
            "summary": "Generated documentation",
            "description": "Documentation could not be fully parsed",
            "args": [],
            "returns": {"type": "Any", "description": "Unknown return value"},
            "raises": [],
            "complexity": 1
        }
        
        # Only validate and copy critical string fields
        for key in ['summary', 'description']:
            if key in partial_data and isinstance(partial_data[key], str):
                fallback[key] = partial_data[key]
        
        # Copy complexity if valid
        if 'complexity' in partial_data and isinstance(partial_data['complexity'], int):
            fallback['complexity'] = partial_data['complexity']
            
        return fallback

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_function = load_schema('function_tools_schema')

    def _create_function_calling_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Create the initial prompt for function calling."""
        return (
            "You are a highly skilled Python documentation expert. Generate comprehensive "
            "documentation following these specific requirements:\n\n"
            "1. DOCSTRING FORMAT:\n"
            "- Use Google-style docstrings\n"
            "- Include complexity scores for all functions and classes\n"
            "- Add warning emoji (⚠️) for complexity scores > 10\n"
            "- Document all parameters with their types\n"
            "- Document return values with types\n"
            "- Document raised exceptions\n\n"
            "2. DOCUMENTATION STRUCTURE:\n"
            "- Start with a clear summary line\n"
            "- Provide detailed description\n"
            "- List and explain all parameters\n"
            "- Describe return values\n"
            "- Document exceptions/errors\n"
            "- Include complexity metrics\n\n"
            "3. Generate a JSON object with the following structure:\n"
            "{\n"
            "  'summary': '<summary>',\n"
            "  'description': '<detailed_description>',\n"
            "  'args': [{\n"
            "    'name': '<arg_name>',\n"
            "    'type': '<arg_type>',\n"
            "    'description': '<arg_description>'\n"
            "  }, ...],\n"
            "  'returns': { 'type': '<return_type>', 'description': '<return_description>' },\n"
            "  'raises': [{\n"
            "    'exception': '<exception_name>',\n"
            "    'description': '<exception_description>'\n"
            "  }, ...],\n"
            "  'complexity': <complexity_score>\n"
            "}\n\n"
            "The code to document is:\n"
            "```python\n"
            f"{source_code}\n"
            "```\n"
        )

    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            if hasattr(self, 'client'):
                await self.client.close()
            if self.cache:
                await self.cache.close()
            if hasattr(self, 'metrics_calculator') and self.metrics_calculator:
                await self.metrics_calculator.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()