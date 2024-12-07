"""
This module provides classes and functions for handling AI interactions, processing source code,
generating dynamic prompts, and integrating AI-generated documentation back into the source code.

Classes:
    CustomJSONEncoder: Custom JSON encoder that can handle sets and other non-serializable types.
    AIInteractionHandler: Handles AI interactions for generating enriched prompts and managing responses.

Functions:
    serialize_for_logging(obj: Any) -> str: Safely serialize any object for logging purposes.
"""

import ast
import asyncio
import json
import re
import types
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncAzureOpenAI

from api.api_client import APIClient
from api.token_management import TokenManager
from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.logger import LoggerSetup
from core.markdown_generator import MarkdownGenerator
from core.metrics import Metrics
from core.response_parsing import ParsedResponse, ResponseParsingService
from core.schema_loader import load_schema
from core.types import ExtractionContext, ExtractionResult
from exceptions import ConfigurationError, ProcessingError

logger = LoggerSetup.get_logger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle sets and other non-serializable types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (ast.AST, types.ModuleType)):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in obj.__dict__.items()
                if isinstance(key, str) and not key.startswith("_")
            }
        return super().default(obj)


def serialize_for_logging(obj: Any) -> str:
    """Safely serialize any object for logging purposes."""
    try:
        return json.dumps(obj, cls=CustomJSONEncoder, indent=2)
    except Exception as e:
        return f"Error serializing object: {str(e)}\nObject repr: {repr(obj)}"


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
        """Initialize the AIInteractionHandler."""
        self.logger = logger
        self.config = config or AzureOpenAIConfig().from_env()
        self.cache: Cache | None = cache or self.config.cache
        self.token_manager: TokenManager = token_manager or TokenManager()
        self.metrics: Metrics = metrics or Metrics()
        self.response_parser: ResponseParsingService = response_parser or ResponseParsingService()
        self.docstring_processor = DocstringProcessor(metrics=self.metrics)
        self.docstring_schema: dict[str, Any] = docstring_schema or load_schema()

    def _truncate_response(
        self, response: Union[str, Dict[str, Any]], length: int = 200
    ) -> str:
        """
        Safely truncate a response for logging.

        Args:
            response: The response to truncate (either string or dictionary)
            length: Maximum length of the truncated response

        Returns:
            str: Truncated string representation of the response
        """
        try:
            if isinstance(response, dict):
                json_str = json.dumps(response, indent=2)
                return (json_str[:length] + "...") if len(json_str) > length else json_str
            elif isinstance(response, str):
                return (response[:length] + "...") if len(response) > length else response
            else:
                str_response = str(response)
                return (str_response[:length] + "...") if len(str_response) > length else str_response
        except Exception as e:
            return f"<Error truncating response: {str(e)}>"

    async def process_code(self, source_code: str) -> Optional[Dict[str, Any]]:
        """Process the source code to extract metadata, interact with the AI, and integrate responses."""
        try:
            tree: ast.AST = ast.parse(source_code)
            context = ExtractionContext()
            context.source_code = source_code
            context.tree = tree

            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                match = re.search(r"Module:?\s*([^\n\.]+)", module_docstring)
                if match:
                    context.module_name = match.group(1).strip()

            extractor = CodeExtractor(context)
            extraction_result: ExtractionResult | None = await extractor.extract_code(source_code)
            if not extraction_result:
                self.logger.error("Failed to extract code elements")
                return None

            extracted_info = {
                "module_docstring": extraction_result.module_docstring or "",
                "classes": [cls.to_dict() for cls in (extraction_result.classes or [])],
                "functions": [func.to_dict() for func in (extraction_result.functions or [])],
                "dependencies": extraction_result.dependencies or {},
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            self.logger.debug("Generated prompt for AI")

            ai_response: str | dict[str, Any] = await self._interact_with_ai(prompt)
            self.logger.debug(f"Received AI response: {self._truncate_response(ai_response)}")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                ai_response, expected_format="docstring"
            )

            if not parsed_response.validation_success:
                self.logger.error(f"Failed to validate AI response. Errors: {parsed_response.errors}")
                return None

            updated_code, documentation = await self._integrate_ai_response(
                parsed_response.content, extraction_result
            )

            if not updated_code or not documentation:
                self.logger.error("Integration produced empty results")
                return None

            return {"code": updated_code, "documentation": documentation}

        except (SyntaxError, ValueError, TypeError) as e:
            self.logger.error(f"Error processing code: {e}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error processing code: {e}", exc_info=True)
            return None

    async def _integrate_ai_response(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> Tuple[str, str]:
        """Integrate the AI response into the source code and update the documentation."""
        try:
            self.logger.debug("Starting AI response integration")
            ai_response = self._ensure_required_fields(ai_response)
            processed_response = self._create_processed_response(ai_response)

            integration_result = self._process_docstrings(processed_response, extraction_result.source_code)
            if not integration_result:
                raise ProcessingError("Docstring integration failed")

            code = integration_result.get("code", "")
            if not isinstance(code, str):
                raise ProcessingError("Expected 'code' to be a string in integration result")

            documentation = self._generate_markdown_documentation(ai_response, extraction_result)
            return code, documentation

        except Exception as e:
            self.logger.error(f"Error integrating AI response: {e}", exc_info=True)
            raise ProcessingError(f"AI response integration failed: {str(e)}") from e

    def _generate_markdown_documentation(
        self, ai_response: Dict[str, Any], extraction_result: ExtractionResult
    ) -> str:
        """Generate markdown documentation from AI response and extraction result."""
        markdown_gen = MarkdownGenerator()
        markdown_context: Dict[str, Any] = {
            "module_name": extraction_result.module_name,
            "file_path": extraction_result.file_path,
            "description": ai_response.get("description", ""),
            "classes": extraction_result.classes,
            "functions": extraction_result.functions,
            "constants": extraction_result.constants,
            "source_code": extraction_result.source_code,
            "ai_documentation": ai_response,
        }
        return markdown_gen.generate(markdown_context)

    def _ensure_required_fields(self, ai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the AI response has the required fields, and repair if necessary."""
        required_fields = ["summary", "description", "args", "returns", "raises"]
        if not all(field in ai_response for field in required_fields):
            missing = [f for f in required_fields if f not in ai_response]
            self.logger.error(f"AI response missing required fields: {missing}")

            for field in missing:
                if field == "args":
                    ai_response["args"] = []
                elif field == "returns":
                    ai_response["returns"] = {"type": "None", "description": ""}
                elif field == "raises":
                    ai_response["raises"] = []
                else:
                    ai_response[field] = ""
        return ai_response

    def _create_processed_response(self, ai_response: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Create a list with the processed response."""
        return [
            {
                "name": "__module__",  # Use module-level docstring
                "docstring": ai_response,
                "type": "Module",
            }
        ]

    def _process_docstrings(self, processed_response: List[Dict[str, Union[str, Dict[str, Any]]]], source_code: str) -> Dict[str, Any]:
        """Process the docstrings using DocstringProcessor."""
        integration_result = self.docstring_processor.process_batch(processed_response, source_code)
        if not integration_result:
            raise ProcessingError("Docstring processor returned no results")
        self.logger.debug("Successfully processed docstrings")
        return integration_result

    async def _interact_with_ai(self, prompt: str) -> str | dict[str, Any]:
        """Interact with the AI model to generate responses."""
        try:
            request_params: dict[str, Any] = await self.token_manager.validate_and_prepare_request(prompt)
            request_params['max_tokens'] = 1000

            self.logger.debug("Sending request to AI")

            system_prompt = """You are a Python documentation expert. Generate complete docstrings in Google format.
Return ONLY a JSON object with this structure, no other text:
{
"summary": "Brief one-line summary",
"description": "Detailed multi-line description",
"args": [{"name": "param_name", "type": "param_type", "description": "param description"}],
"returns": {"type": "return_type", "description": "what is returned"},
"raises": [{"exception": "ErrorType", "description": "when this error occurs"}]
}
Include only the JSON object in the response without any additional text or formatting."""

            response = await self.client.chat.completions.create(
                model=self.config.deployment_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request_params['max_tokens'],
                temperature=request_params.get('temperature', 0.7)
            )

            if not response.choices:
                raise ProcessingError("AI response contained no choices")

            response_content = response.choices[0].message.content
            if not response_content:
                raise ProcessingError("AI response content is empty")

            self.logger.debug("Raw response received from AI")

            try:
                response_json: Dict[str, Any] = json.loads(response_content)
                self.logger.debug("Successfully parsed response as JSON")
                return response_json
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse AI response as JSON: {e}")

                # Try to extract JSON from the response
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        response_json = json.loads(json_str)
                        self.logger.debug("Successfully extracted and parsed JSON from response")
                        return response_json
                    except json.JSONDecodeError as e2:
                        self.logger.error(f"Failed to parse extracted JSON: {e2}")

                # Return the raw response if JSON parsing fails
                return response_content

        except (json.JSONDecodeError, ProcessingError) as e:
            self.logger.error(f"Error during AI interaction: {e}", exc_info=True)
            raise ProcessingError(f"AI interaction failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during AI interaction: {e}", exc_info=True)
            raise

    async def create_dynamic_prompt(self, extracted_info: dict[str, str | list[dict[str, Any]] | dict[str, Any]]) -> str:
        """Create a dynamic prompt for the AI model."""
        try:
            self.logger.debug("Creating dynamic prompt")
            self.logger.debug(f"Extracted info: {serialize_for_logging(extracted_info)}")

            prompt_parts: List[str] = [
                "Generate a complete Python documentation structure as a single JSON object.\n\n",
                "Required JSON structure:\n",
                "{\n",
                '  "summary": "One-line summary of the code",\n',
                '  "description": "Detailed description of functionality",\n',
                '  "args": [{"name": "param1", "type": "str", "description": "param description"}],\n',
                '  "returns": {"type": "ReturnType", "description": "return description"},\n',
                '  "raises": [{"exception": "ValueError", "description": "error description"}]\n',
                "}\n\n",
                "Code Analysis:\n"
            ]

            if extracted_info.get("module_docstring"):
                prompt_parts.append(f"Current Module Documentation:\n{extracted_info['module_docstring']}\n\n")

            if extracted_info.get("classes"):
                prompt_parts.append("Classes:\n")
                for cls in extracted_info["classes"]:
                    prompt_parts.append(f"- {cls['name']}\n")
                    if cls.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {cls['docstring']}\n")
                    if cls.get("methods"):
                        prompt_parts.append("  Methods:\n")
                        for method in cls["methods"]:
                            prompt_parts.append(f"    - {method['name']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("functions"):
                prompt_parts.append("Functions:\n")
                for func in extracted_info["functions"]:
                    prompt_parts.append(f"- {func['name']}\n")
                    if func.get("docstring"):
                        prompt_parts.append(f"  Current docstring: {func['docstring']}\n")
                prompt_parts.append("\n")

            if extracted_info.get("dependencies"):
                prompt_parts.append("Dependencies:\n")
                for dep_type, deps in extracted_info["dependencies"].items():
                    if deps:
                        prompt_parts.append(f"- {dep_type}: {', '.join(deps)}\n")
                prompt_parts.append("\n")

            prompt_parts.append(
                "Based on the above code analysis, generate a single JSON object with "
                "comprehensive documentation following the required structure. Include only "
                "the JSON object in your response, no other text."
            )

            prompt: str = "".join(prompt_parts)
            self.logger.debug(f"Generated prompt: {prompt[:500]}...")
            return prompt

        except Exception as e:
            self.logger.error(f"Error creating prompt: {e}", exc_info=True)
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
        exceptions: Optional[List[Dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Generate a docstring for a function or class.

        Args:
            func_name: Name of the function or class
            is_class: Whether the target is a class
            params: List of parameters with their types and descriptions
            return_type: Return type of the function
            complexity_score: Complexity score of the function
            existing_docstring: Existing docstring to enhance
            decorators: List of decorators applied to the function
            exceptions: List of exceptions raised by the function

        Returns:
            dict: Generated docstring content
        """
        params = params or []
        decorators = decorators or []
        exceptions = exceptions or []

        try:
            extracted_info: Dict[str, Any] = {
                "name": func_name,
                "params": params,
                "returns": {"type": return_type},
                "complexity": complexity_score,
                "existing_docstring": existing_docstring,
                "decorators": decorators,
                "raises": exceptions,
                "is_class": is_class,
            }

            prompt: str = await self.create_dynamic_prompt(extracted_info)
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                stop=["END"],
            )
            response_content: str | None = response.choices[0].message.content
            if response_content is None:
                raise ProcessingError("AI response content is empty")

            parsed_response: ParsedResponse = await self.response_parser.parse_response(
                response_content
            )
            self.logger.info("Generated docstring for %s", func_name)
            return parsed_response.content

        except json.JSONDecodeError as e:
            self.logger.error(
                "JSON decoding error while generating docstring for %s: %s",
                func_name,
                e,
            )
            raise
        except ProcessingError as e:
            self.logger.error(
                "Processing error while generating docstring for %s: %s", func_name, e
            )
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error generating docstring for %s: %s", func_name, e
            )
            raise

    async def _verify_deployment(self) -> bool:
        """
        Verify that the configured deployment exists and is accessible.
        """
        try:
            test_params: Dict[str, Any] = {
                "model": self.config.deployment_id,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            }
            self.logger.debug(f"Verifying deployment with parameters: {test_params}")
            response = await self.client.chat.completions.create(**test_params)
            self.logger.debug(f"Deployment verification response: {response}")
            return True
        except Exception as e:
            self.logger.error(f"Deployment verification failed: {e}", exc_info=True)
            return False

    async def __aenter__(self) -> "AIInteractionHandler":
        """
        Async context manager entry.

        Verifies the deployment configuration and raises a ConfigurationError if the deployment is not accessible.
        """
        if not await self._verify_deployment():
            raise ConfigurationError(
                f"Azure OpenAI deployment '{self.config.deployment_id}' "
                "is not accessible. Please verify your configuration."
            )
        return self

    async def close(self) -> None:
        """Cleanup resources held by AIInteractionHandler."""
        self.logger.debug("Starting cleanup of AIInteractionHandler resources")
        if self.cache is not None:
            await self.cache.close()
            self.logger.debug("Cache resources have been cleaned up")
        self.logger.info("AIInteractionHandler resources have been cleaned up")
