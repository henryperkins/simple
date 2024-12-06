"""Enhanced AI interaction handler to integrate enriched metadata in prompt generation."""

from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.types import ExtractionResult
from core.extraction.code_extractor import CodeExtractor
from core.response_parsing import ResponseParsingService
from core.docstring_processor import DocstringProcessor
from exceptions import ProcessingError
from openai import AsyncAzureOpenAI
import asyncio
import json

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    """Handles AI interactions for generating enriched prompts and managing responses."""

    def __init__(self, metrics: Optional[Metrics] = None, azure_config: Optional[Dict[str, str]] = None, docstring_schema: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the AI interaction handler."""
        self.logger = logger
        self.metrics = metrics or Metrics()
        self.response_parser = ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema = docstring_schema or self.load_schema("docstring_schema")
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": self.docstring_schema['schema']
            }
        }

        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_config['endpoint'],
            api_key=azure_config['api_key'],
            api_version=azure_config['api_version'],
        ) if azure_config else None

    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load the schema from a JSON file."""
        with open(f"{schema_name}.json", "r") as file:
            return json.load(file)

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
        """
        Process source code by extracting relevant information, generating enriched documentation, and integrating AI responses.

        Args:
            source_code (str): The source code to be analyzed.

        Returns:
            Dict[str, Any]: The resulting updated source code and documentation.
        """
        try:
            # Step 1: Extract metadata using CodeExtractor
            extractor = CodeExtractor()
            extraction_result = await extractor.extract_code(source_code)
            
            # Step 2: Generate a prompt using extracted metadata
            extracted_info = {
                "module_docstring": extraction_result.module_docstring,
                "classes": [cls.__dict__ for cls in extraction_result.classes],
                "functions": [func.__dict__ for func in extraction_result.functions],
                "dependencies": extraction_result.dependencies,
            }
            prompt = self.create_dynamic_prompt(extracted_info)
            
            # Step 3: Interact with AI to generate docstrings
            ai_response = await self._interact_with_ai(prompt)
            
            # Step 4: Parse AI response and validate
            parsed_response = await self.response_parser.parse_response(ai_response, expected_format="docstring")
            if not parsed_response.validation_success:
                self.logger.error("Validation failed for AI response.")
                raise ProcessingError("Failed to validate AI response.")
            
            # Step 5: Process the AI response and integrate docstrings
            updated_code, updated_documentation = await self._integrate_ai_response(parsed_response.content, extraction_result)
            
            return {'code': updated_code, 'documentation': updated_documentation}
        except Exception as e:
            self.logger.error("Error processing code: %s", e)
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
            if not self.client:
                raise ValueError("Azure OpenAI client is not initialized.")

            self.logger.info("Interacting with AI to generate response.")
            response = await self.client.chat.completions.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                stop=["END"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            self.logger.error("Error during AI interaction: %s", e)
            raise ProcessingError("AI interaction failed.") from e

    async def _integrate_ai_response(self, ai_response: str, extraction_result: ExtractionResult) -> Tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.
        
        Returns:
            Tuple[str, str]: The updated source code and documentation.
        """
        try:
            # Validate AI response
            parsed_content = await self.response_parser.parse_response(ai_response, expected_format="docstring")
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
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.7,
                "stop": ["END"]
            }
            response = await self.client.chat.completions.create(**request_params)
            parsed_response = await self.response_parser.parse_response(response)
            self.logger.info("Generated docstring for %s", func_name)
            return parsed_response.content

        except Exception as e:
            self.logger.error("Error generating docstring for %s: %s", func_name, e)
            raise
