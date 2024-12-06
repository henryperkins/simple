"""
Module ai_interaction.py: Handles interactions with an AI model for generating enriched prompts,
managing responses, and integrating AI-generated documentation into source code.

This module provides the AIInteractionHandler class, which processes source code by extracting
metadata, generates prompts for the AI model, and integrates the AI's responses back into
the source code as documentation. It utilizes services such as TokenManager, Cache,
CodeExtractor, and ResponseParsingService to manage tokens, caching, code extraction,
and response parsing respectively. The primary goal is to enhance code documentation using
AI-generated content while maintaining code maintainability and readability.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncAzureOpenAI

from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.response_parsing import ResponseParsingService
from core.schema_loader import load_schema
from api.api_client import APIClient
from core.types import ExtractionResult
from exceptions import ProcessingError, ConfigurationError

logger = LoggerSetup.get_logger(__name__)


class AIInteractionHandler:
    """
    Handles AI interactions for generating enriched prompts and managing responses.

    This class is responsible for processing source code, generating dynamic prompts for
    the AI model, handling AI interactions, parsing AI responses, and integrating the
    AI-generated documentation back into the source code. It ensures that the generated
    documentation is validated and integrates seamlessly with the existing codebase.

    Attributes:
        config (AzureOpenAIConfig): Configuration for the Azure OpenAI client.
        cache (Cache): Cache to store and retrieve processed data.
        token_manager (TokenManager): Manages token usage and estimates.
        metrics (Metrics): Collects and reports metrics.
        response_parser (ResponseParsingService): Parses and validates AI responses.
        docstring_processor (DocstringProcessor): Processes docstrings and integrates them into code.
        docstring_schema (Dict[str, Any]): Schema for docstring validation.
        docstring_tool (Dict[str, Any]): Tool specification for generating docstrings.
        client (AsyncAzureOpenAI): Asynchronous Azure OpenAI client for AI interactions.
    """
    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
        docstring_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = logger
        self.config = config or AzureOpenAIConfig.from_env()
        self.cache = cache or self.config.cache
        self.token_manager = token_manager or TokenManager()
        self.metrics = metrics or Metrics()
        self.response_parser = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema = docstring_schema or load_schema("docstring_schema")
        
        # Initialize API client
        self.api_client = APIClient(
            config=self.config,
            response_parser=self.response_parser,
            token_manager=self.token_manager
        )
        # Get the client from APIClient
        self.client = self.api_client.get_client()

    async def create_dynamic_prompt(self, extracted_info: Dict[str, Any], context: str = "") -> str:
        try:
            prompt_parts = []
            prompt_parts.append(
                "You are an AI tasked with enhancing documentation for the provided code. "
                "Below is the extracted metadata:\n\n"
            )

            if "module_docstring" in extracted_info:
                prompt_parts.append(f"Module Docstring: {extracted_info['module_docstring']}\n\n")

            # Add classes to the prompt
            if "classes" in extracted_info and extracted_info["classes"]:
                for cls in extracted_info["classes"]:
                    if isinstance(cls, dict):
                        name = cls.get('name', 'Unknown')
                        docstring = cls.get('docstring', '')
                        methods = cls.get('methods', [])
                        bases = cls.get('bases', [])
                        metrics = cls.get('metrics', {})
                    else:  # Handle class objects
                        name = getattr(cls, 'name', 'Unknown')
                        docstring = getattr(cls, 'docstring', '')
                        methods = getattr(cls, 'methods', [])
                        bases = getattr(cls, 'bases', [])
                        metrics = getattr(cls, 'metrics', {})

                    # Handle coroutine methods
                    if asyncio.iscoroutine(methods):
                        methods = await methods

                    prompt_parts.append(f"Class: {name}\n")
                    prompt_parts.append(f"  Description: {docstring}\n")
                    method_names = []
                    for m in methods:
                        if asyncio.iscoroutine(m):
                            m = await m
                        method_names.append(getattr(m, 'name', str(m)))
                    prompt_parts.append(f"  Methods: {method_names}\n")
                    prompt_parts.append(f"  Bases: {bases}\n")
                    maintainability = metrics.get('maintainability_index', 'N/A') if isinstance(metrics, dict) else 'N/A'
                    prompt_parts.append(f"  Maintainability Index: {maintainability}\n\n")

            # Add functions to the prompt
            if "functions" in extracted_info and extracted_info["functions"]:
                for func in extracted_info["functions"]:
                    if isinstance(func, dict):
                        name = func.get('name', 'Unknown')
                        docstring = func.get('docstring', '')
                        metrics = func.get('metrics', {})
                    else:  # Handle ExtractedFunction objects
                        name = getattr(func, 'name', 'Unknown')
                        docstring = getattr(func, 'docstring', '')
                        metrics = getattr(func, 'metrics', {})

                    prompt_parts.append(f"Function: {name}\n")
                    prompt_parts.append(f"  Description: {docstring}\n")
                    if isinstance(metrics, dict):
                        cyclomatic = metrics.get('cyclomatic_complexity', 'N/A')
                        cognitive = metrics.get('cognitive_complexity', 'N/A')
                    else:
                        cyclomatic = 'N/A'
                        cognitive = 'N/A'
                    prompt_parts.append(f"  Complexity: {cyclomatic}\n")
                    prompt_parts.append(f"  Cognitive Complexity: {cognitive}\n\n")

            # Include any dependencies if available
            if "dependencies" in extracted_info and extracted_info["dependencies"]:
                prompt_parts.append("Dependencies:\n")
                if isinstance(extracted_info["dependencies"], dict):
                    for dep, details in extracted_info["dependencies"].items():
                        prompt_parts.append(f"- {dep}\n")
                        if isinstance(details, dict):
                            for key, value in details.items():
                                prompt_parts.append(f"  {key}: {value}\n")
                else:
                    for dep in extracted_info["dependencies"]:
                        prompt_parts.append(f"- {dep}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Please generate or improve docstrings for all extracted classes, functions, and methods."
            )

            prompt = "".join(prompt_parts)
            self.logger.debug("Generated prompt for AI interaction: %s", prompt)
            return prompt
        except Exception as e:
            self.logger.error("Error generating prompt: %s", e)
            raise

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        try:
            # Extract metadata
            extractor = CodeExtractor()
            extraction_result = await extractor.extract_code(source_code)

            if not extraction_result:
                self.logger.error("Failed to extract code elements")
                return None

            # Build extracted info safely
            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [
                    {
                        "name": cls.name,
                        "docstring": cls.docstring,
                        "methods": cls.methods,
                        "bases": cls.bases,
                        "metrics": cls.metrics,
                    }
                    for cls in (extraction_result.classes or [])
                ],
                "functions": [
                    {
                        "name": func.name,
                        "docstring": func.docstring,
                        "metrics": func.metrics,
                        "raises": func.raises,
                    }
                    for func in (extraction_result.functions or [])
                ],
                "dependencies": extraction_result.dependencies or {
                    "Monitoring Module": {
                        "description": "Provides system monitoring and performance tracking for Azure OpenAI operations.",
                        "imports": ["asyncio", "defaultdict", "datetime", "timedelta", "Dict", "Any", "Optional", "List", "psutil", "LoggerSetup", "MetricsCollector", "TokenManager"]
                    }
                },
            }

            # Generate and process - Fix here
            prompt = await self.create_dynamic_prompt(extracted_info)  # Add await here
            if not isinstance(prompt, str):
                self.logger.error("Generated prompt is not a string")
                return None
                
            ai_response = await self._interact_with_ai(prompt)
            
            # Continue with the rest of the code...
            parsed_response = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                self.logger.error("Failed to validate AI response")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )
            return {"code": updated_code, "documentation": documentation}

        except Exception as e:
            self.logger.error(f"Error processing code: {e}", exc_info=True)
            return None

    async def _interact_with_ai(self, prompt: str) -> str:
        """Interact with the AI model to generate responses."""
        try:
            request_tokens = self.token_manager.estimate_tokens(prompt)
            request_params = await self.token_manager.validate_and_prepare_request(prompt)
            
            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=request_params.get("max_tokens", 1000),
                temperature=request_params.get("temperature", 0.7)
            )

            response_content = response.choices[0].message.content
            response_tokens = self.token_manager.estimate_tokens(response_content)
            self.token_manager.track_request(request_tokens, response_tokens)
            
            return response_content
        except Exception as e:
            self.logger.error(f"Error during AI interaction: {e}")
            raise ProcessingError("AI interaction failed.") from e
        
    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.

        Args:
            ai_response (Dict[str, Any]): The parsed AI response containing documentation updates.
            extraction_result (ExtractionResult): The extracted code information.

        Returns:
            Tuple[str, str]: A tuple containing the updated source code and documentation.

        Raises:
            ProcessingError: If the AI response validation fails.

        Example:
            updated_code, documentation = await self._integrate_ai_response(ai_response, extraction_result)
        """
        try:
            # Validate AI response
            parsed_content = await self.response_parser.parse_response(
                json.dumps(ai_response), expected_format="docstring"
            )
            if not parsed_content.validation_success:
                raise ProcessingError("Failed to validate AI response.")

            # Process and integrate the AI response into the code
            integration_result = self.docstring_processor.process_batch(
                [parsed_content.content], extraction_result.source_code
            )
            return integration_result["code"], integration_result["documentation"]
        except Exception as e:
            self.logger.error("Error integrating AI response: %s", e)
            raise

    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name (str): Name of the function or class.
            params (Optional[List[Dict[str, Any]]]): List of parameter information.
            return_type (str, optional): Return type of the function. Defaults to "Any".
            complexity_score (int, optional): Complexity score of the code. Defaults to 0.
            existing_docstring (str, optional): Existing docstring if any. Defaults to "".
            decorators (Optional[List[str]]): List of decorators.
            exceptions (Optional[List[Dict[str, str]]]): List of exceptions that can be raised.
            is_class (bool, optional): Whether this is a class docstring. Defaults to False.

        Returns:
            Dict[str, Any]: The generated docstring data.

        Raises:
            Exception: If there is an error generating the docstring.

        Example:
            docstring_data = await ai_handler.generate_docstring("my_function")
        """
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            extracted_info = {
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            }

            prompt = self.create_dynamic_prompt(extracted_info)
            request_params = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
                "stop": ["END"],
            }
            response = await self.client.chat.completions.create(**request_params)
            parsed_response = await self.response_parser.parse_response(response)
            self.logger.info("Generated docstring for %s", func_name)
            return parsed_response.content

        except Exception as e:
            self.logger.error("Error generating docstring for %s: %s", func_name, e)
            raise
    async def _verify_deployment(self) -> bool:
        """Verify that the configured deployment exists and is accessible."""
        try:
            # Make a minimal test request
            test_params = {
                "model": self.config.deployment_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            await self.client.chat.completions.create(**test_params)
            return True
        except Exception as e:
            self.logger.error(f"Deployment verification failed: {e}")
            return False

    async def __aenter__(self) -> 'AIInteractionHandler':
        """Async context manager entry."""
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self
    async def close(self) -> None:
        """
        Cleanup resources held by AIInteractionHandler.

        This method should be called to properly clean up any resources when the
        AIInteractionHandler is no longer needed, such as closing cache connections.

        Example:
            await ai_handler.close()
        """
        # Example: Close cache connection if necessary
        if self.cache:
            await self.cache.close()
        # Add any additional cleanup steps here
        self.logger.info("AIInteractionHandler resources have been cleaned up")