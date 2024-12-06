import asyncio
import json
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
from core.types import ExtractionResult, ExtractedFunction, ExtractedClass
from exceptions import ProcessingError, ConfigurationError

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    """
    Handles AI interactions for generating enriched prompts and managing responses.

    This class is responsible for processing source code, generating dynamic prompts for
    the AI model, handling AI interactions, parsing AI responses, and integrating the
    AI-generated documentation back into the source code. It ensures that the generated
    documentation is validated and integrates seamlessly with the existing codebase.
    """

    def __init__(
        self,
        config: AzureOpenAIConfig | None = None,
        cache: Cache | None = None,
        token_manager: TokenManager | None = None,
        response_parser: ResponseParsingService | None = None,
        metrics: Metrics | None = None,
        docstring_schema: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the AIInteractionHandler.
        """
        self.logger = logger
        self.config = config or AzureOpenAIConfig.from_env()
        self.cache = cache or self.config.cache
        self.token_manager = token_manager or TokenManager()
        self.metrics = metrics or Metrics()
        self.response_parser = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: Dict[str, Any] = docstring_schema or load_schema("docstring_schema")
        
        # Initialize API client
        self.api_client = APIClient(
            config=self.config,
            response_parser=self.response_parser,
            token_manager=self.token_manager
        )
        self.client = self.api_client.get_client()

    async def create_dynamic_prompt(self, extracted_info: dict[str, Any], context: str = "") -> str:
        """
        Create a dynamic prompt for the AI model based on extracted code information.
        """
        try:
            prompt_parts = [
                "You are an AI tasked with enhancing documentation for the provided code. "
                "Below is the extracted metadata:\n\n"
            ]

            def serialize_extracted_element(element: Any) -> Dict[str, Any]:
                """Helper function to serialize ExtractedFunction/ExtractedClass."""
                if isinstance(element, (ExtractedFunction, ExtractedClass)):
                    return element.to_dict()
                return element

            def add_elements_to_prompt(element_type: str, elements: List[Any]):
                if elements:
                    prompt_parts.append(f"{element_type.capitalize()}s:\n")
                    for element in elements:
                        serialized_element = serialize_extracted_element(element)
                        for key, value in serialized_element.items():
                            if isinstance(value, list):
                                if key == 'methods':
                                    value = [serialize_extracted_element(m) for m in value]
                                    value = [m.get('name') for m in value if isinstance(m, dict) and 'name' in m]
                                elif key == 'raises':
                                    value = [r.get('exception') for r in value if isinstance(r, dict) and 'exception' in r]
                            prompt_parts.append(f"- {key}: {value}\n")
                        prompt_parts.append("\n")

            add_elements_to_prompt("Class", extracted_info.get("classes", []))
            add_elements_to_prompt("Function", extracted_info.get("functions", []))

            if "dependencies" in extracted_info and extracted_info["dependencies"]:
                prompt_parts.append("Dependencies:\n")
                for dep, details in extracted_info["dependencies"].items():
                    prompt_parts.append(f"- {dep}\n")
                    if isinstance(details, dict):
                        for key, value in details.items():
                            prompt_parts.append(f"  {key}: {value}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Please generate or improve docstrings for all extracted classes, functions, and methods."
            )

            prompt = "".join(prompt_parts)
            self.logger.debug("Generated prompt for AI interaction: %s", prompt)
            return prompt

        except Exception as e:
            self.logger.error("Error generating prompt: %s", e, exc_info=True)
            raise

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """
        Process the source code to extract metadata, interact with the AI, and integrate responses.
        """
        try:
            extractor = CodeExtractor()
            extraction_result = await extractor.extract_code(source_code)

            if not extraction_result:
                self.logger.error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {}
            }

            prompt = await self.create_dynamic_prompt(extracted_info)

            ai_response = await self._interact_with_ai(prompt)
            parsed_response = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response["validation_success"]:
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
        """
        Interact with the AI model to generate responses.
        """
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
            if response_content is None:
                raise ProcessingError("AI response content is None.")

            # Parse and validate the response using ResponseParsingService
            parsed_response = await self.response_parser.parse_response(
                response_content, 
                expected_format="docstring"
            )
            if not parsed_response.validation_success:
                raise ProcessingError("Failed to validate AI response.")

            # Track token usage
            response_tokens = self.token_manager.estimate_tokens(response_content)
            self.token_manager.track_request(request_tokens, response_tokens)
            
            return response_content
            
        except Exception as e:
            self.logger.error(f"Error during AI interaction: {e}")
            raise ProcessingError("AI interaction failed.") from e
        
    async def _integrate_ai_response(
        self, ai_response: dict[str, Any], extraction_result: ExtractionResult
    ) -> tuple[str, str]:
        """
        Integrate the AI response into the source code and update the documentation.
        """
        try:
            # Use parse_response instead of direct validation
            parsed_response = await self.response_parser.parse_response(
                ai_response, 
                expected_format="docstring"
            )
            if not parsed_response.validation_success:
                raise ProcessingError("Failed to validate AI response.")

            integration_result = self.docstring_processor.process_batch(
                [parsed_response.content], extraction_result.source_code
            )
            return integration_result["code"], integration_result["documentation"]
        except Exception as e:
            self.logger.error("Error integrating AI response: %s", e)
            raise
    
    async def generate_docstring(
        self,
        func_name: str,
        is_class: bool,
        params: List[Dict[str, Any]] | None = None,
        return_type: str = "Any",
        complexity_score: int = 0,
        existing_docstring: str = "",
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a docstring for a function or class.
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

            prompt = await self.create_dynamic_prompt(extracted_info)
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
        """
        Verify that the configured deployment exists and is accessible.
        """
        try:
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
        """
        Async context manager entry.
        """
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self

    async def close(self) -> None:
        """
        Cleanup resources held by AIInteractionHandler.
        """
        if self.cache:
            await self.cache.close()
        self.logger.info("AIInteractionHandler resources have been cleaned up")