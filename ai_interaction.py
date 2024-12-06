"""Enhanced AI interaction handler to integrate enriched metadata in prompt generation."""

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
from core.types import ExtractionResult
from exceptions import ProcessingError

logger = LoggerSetup.get_logger(__name__)


class AIInteractionHandler:
    """Handles AI interactions for generating enriched prompts and managing responses."""

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
        docstring_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the AI interaction handler."""
        self.logger = logger
        self.config = config or AzureOpenAIConfig.from_env()
        self.cache = cache or self.config.cache
        self.token_manager = token_manager or TokenManager()
        self.metrics = metrics or Metrics()
        self.response_parser = ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema = docstring_schema or load_schema("docstring_schema")
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": self.docstring_schema['schema']
            }
        }

        # Initialize the Azure OpenAI client using the configuration
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
        )

    def create_dynamic_prompt(self, extracted_info: Dict[str, Any], context: str = "") -> str:
        """Create a dynamic prompt based on extracted code information and context."""
        try:
            prompt = """You are an AI tasked with enhancing documentation for the provided code. Below is the extracted metadata:
            
            """
            if "module_docstring" in extracted_info:
                prompt += f"Module Docstring: {extracted_info['module_docstring']}\n\n"
            
            # Add classes to the prompt
            if "classes" in extracted_info:
                for cls in extracted_info["classes"]:
                    prompt += f"Class: {cls['name']}\n"
                    prompt += f"  Description: {cls['docstring']}\n"
                    prompt += f"  Methods: {[method['name'] for method in cls['methods']]}\n"
                    prompt += f"  Bases: {cls['bases']}\n"
                    prompt += f"  Maintainability Index: {cls['metrics']['maintainability_index']}\n\n"
            
            # Add functions to the prompt
            if "functions" in extracted_info:
                for func in extracted_info["functions"]:
                    prompt += f"Function: {func['name']}\n"
                    prompt += f"  Description: {func['docstring']}\n"
                    prompt += f"  Complexity: {func['metrics']['cyclomatic_complexity']}\n"
                    prompt += f"  Cognitive Complexity: {func['metrics']['cognitive_complexity']}\n\n"

            # Include any dependencies if available
            if "dependencies" in extracted_info:
                prompt += "Dependencies:\n"
                for dep in extracted_info["dependencies"]:
                    prompt += f"- {dep}\n"
                prompt += "\n"
            
            prompt += """
            Please generate or improve docstrings for all extracted classes, functions, and methods.
            """
            
            self.logger.debug("Generated prompt for AI interaction: %s", prompt)
            return prompt
        except Exception as e:
            self.logger.error("Error generating prompt: %s", e)
            raise

    async def process_code(self, source_code: str) -> Dict[str, Any]:
        """Process source code and generate documentation."""
        try:
            # Extract metadata 
            extractor = CodeExtractor()
            extraction_result = await extractor.extract_code(source_code)
            
            if not extraction_result:
                raise ProcessingError("Failed to extract code elements")

            # Build extracted info safely
            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [
                    {
                        "name": cls.get("name", "Unknown"),
                        "docstring": cls.get("docstring", ""),
                        "methods": cls.get("methods", []),
                        "bases": cls.get("bases", []),
                        "metrics": cls.get("metrics", {})
                    } for cls in (extraction_result.classes or [])
                ],
                "functions": [
                    {
                        "name": func.get("name", "Unknown"),
                        "docstring": func.get("docstring", ""),
                        "metrics": func.get("metrics", {}),
                        "raises": func.get("raises", [])
                    } for func in (extraction_result.functions or [])
                ],
                "dependencies": extraction_result.dependencies or {}
            }

            # Generate and process
            prompt = self.create_dynamic_prompt(extracted_info)
            ai_response = await self._interact_with_ai(prompt)
            
            parsed_response = await self.response_parser.parse_response(
                ai_response, 
                expected_format="docstring"
            )
            
            if not parsed_response.validation_success:
                raise ProcessingError("Failed to validate AI response")

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content,
                extraction_result
            )
            return {
                "code": updated_code,
                "documentation": documentation
            }

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            raise

    async def _interact_with_ai(self, prompt: str) -> str:
        """
        Interact with AI to generate responses based on a given prompt.

        Args:
            prompt (str): The prompt to send to the AI model.

        Returns:
            str: The AI-generated response.
        """
        try:
            # Use token_manager to estimate tokens if needed
            request_tokens = self.token_manager.estimate_tokens(prompt)
            
            # Prepare the request parameters
            request_params = await self.token_manager.validate_and_prepare_request(prompt)
            
            # Interact with the AI model
            response = await self.client.chat.completions.create(**request_params)
            
            response_content = response.choices[0].message.content if response.choices else ""
            
            # Parse the response using self.response_parser
            parsed_response = await self.response_parser.parse_response(response_content)
            
            # Use token_manager to track usage
            response_tokens = self.token_manager.estimate_tokens(response_content)
            self.token_manager.track_request(request_tokens, response_tokens)
            
            return parsed_response.content
        except Exception as e:
            self.logger.error(f"Error during AI interaction: {e}")
            raise ProcessingError("AI interaction failed.") from e

    async def _integrate_ai_response(self, ai_response: Dict[str, Any], extraction_result: ExtractionResult) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.
        
        Args:
            ai_response (Dict[str, Any]): The parsed AI response containing documentation updates
            extraction_result (ExtractionResult): The extracted code information
        
        Returns:
            Tuple[str, str]: The updated source code and documentation.
        """
        try:
            # Validate AI response
            parsed_content = await self.response_parser.parse_response(json.dumps(ai_response), expected_format="docstring")
            if not parsed_content.validation_success:
                raise ProcessingError("Failed to validate AI response.")

            integration_result = self.docstring_processor.process_batch([parsed_content.content], extraction_result.source_code)
            return integration_result["code"], integration_result["documentation"]
        except Exception as e:
            logger.error("Error integrating AI response: %s", e)
            raise

    async def generate_docstring(
        self,
        func_name: str,
        params: Optional[List[Dict[str, Any]]] = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None,
        is_class: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name: Name of the function or class.
            params: List of parameter information.
            return_type: Return type of the function.
            complexity_score: Complexity score of the code.
            existing_docstring: Existing docstring if any.
            decorators: List of decorators.
            exceptions: List of exceptions that can be raised.
            is_class: Whether this is a class docstring.

        Returns:
            Generated docstring data.
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
                "stop": ["END"]
            }
            response = await self.client.chat.completions.create(
                **request_params
            )
            parsed_response = await self.response_parser.parse_response(response)
            self.logger.info("Generated docstring for %s", func_name)
            return parsed_response.content

        except Exception as e:
            self.logger.error("Error generating docstring for %s: %s", func_name, e)
            raise

    async def close(self) -> None:
        """Cleanup resources held by AIInteractionHandler."""
        # Example: Close cache connection if necessary
        if self.cache:
            await self.cache.close()
        # Add any additional cleanup steps here
        self.logger.info("AIInteractionHandler resources have been cleaned up")
