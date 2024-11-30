"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API for docstring generation, handling token management,
caching, and response processing with structured JSON outputs.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import ast
from dataclasses import dataclass
import openai
from openai import APIError
from jsonschema import validate, ValidationError as JsonValidationError
from core.logger import LoggerSetup
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.config import AzureOpenAIConfig
from core.docstring_processor import DocstringProcessor, DocstringData
from core.code_extraction import CodeExtractor
from docs.docs import DocStringManager
from api.token_management import TokenManager
from api.api_client import APIClient
from exceptions import ValidationError

logger = LoggerSetup.get_logger(__name__)

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    cached: bool = False
    processing_time: float = 0.0


DOCSTRING_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "description": {"type": "string"},
        "args": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type", "description"]
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["type", "description"]
        },
        "raises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "exception": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["exception", "description"]
            }
        }
    },
    "required": ["summary", "description", "args", "returns"]
}


class AIInteractionHandler:
    """Handles AI interactions for docstring generation via Azure OpenAI API."""
    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None,
        config: Optional[AzureOpenAIConfig] = None,
        code_extractor: Optional[CodeExtractor] = None
    ):
        """
        Initialize the AI Interaction Handler.

        Args:
            cache (Optional[Cache]): Cache instance for storing results.
            metrics_collector (Optional[MetricsCollector]): Collector for gathering metrics.
            token_manager (Optional[TokenManager]): Manager for handling API tokens.
            config (Optional[AzureOpenAIConfig]): Configuration for Azure OpenAI.
            code_extractor (Optional[CodeExtractor]): Extractor for code analysis.
        """
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.metrics_collector = metrics_collector or MetricsCollector()
            self.token_manager = token_manager or TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config
            )
            self.client = APIClient(self.config)
            self.docstring_processor = DocstringProcessor()
            self.code_extractor = code_extractor or CodeExtractor()
            self._initialize_tools()
            logger.info("AI Interaction Handler initialized successfully")

            # Set up OpenAI API configuration
            openai.api_type = "azure"
            openai.api_key = self.config.api_key
            openai.api_base = self.config.endpoint
            openai.api_version = self.config.api_version

            # Initialize module state
            self._current_module_tree = None
            self._current_module_docs = {}
            self._current_module = None

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Process source code to generate docstrings."""
        try:
            # Validate token usage
            valid, metrics, message = await self.token_manager.validate_request(source_code)
            if not valid:
                logger.error(f"Token validation failed: {message}")
                return None

            # Initialize module state
            self._current_module_tree = ast.parse(source_code)
            self._current_module_docs = {}
            self._current_module = type('Module', (), {
                '__name__': cache_key.split(':')[1] if cache_key else 'unknown',
                '__doc__': ast.get_docstring(self._current_module_tree) or '',
                '__version__': '0.1.0',
                '__author__': 'Unknown'
            })
            
            # Generate module documentation
            module_docs = await self._generate_documentation(source_code, {}, self._current_module_tree)
            if not module_docs:
                logger.error("Documentation generation failed")
                return None

            # Process classes and functions
            for node in ast.walk(self._current_module_tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    element_docs = await self._generate_documentation(
                        ast.unparse(node), 
                        {"element_type": type(node).__name__}, 
                        node
                    )
                    if element_docs:
                        self._insert_docstring(node, element_docs.content)

            # Generate final output
            updated_code = ast.unparse(self._current_module_tree)

            # Cache results
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, module_docs.content)

            # Track token usage
            self.token_manager.track_request(metrics['prompt_tokens'], metrics['max_completion_tokens'])
            logger.info(f"Tokens used: {metrics['prompt_tokens']} prompt, {metrics['max_completion_tokens']} completion")

            return updated_code, module_docs.content

        except Exception as e:
            logger.error(f"Process code failed: {e}")
            return None

    async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI."""
        try:
            prompt = self._create_function_calling_prompt(source_code, metadata, node)
            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
                tools=[self.docstring_function],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )
            
            if not response:
                return None
                
            message = response.choices[0].message
            if message.tool_calls and message.tool_calls[0].function:
                function_args = message.tool_calls[0].function.arguments
                response_data = json.loads(function_args)
                validate(instance=response_data, schema=DOCSTRING_SCHEMA)
                docstring_data = DocstringData(**response_data)
                
                # Add complexity if node provided
                if node and self.metrics_collector:
                    complexity = self.metrics_collector.calculate_complexity(node)
                    docstring_data.set_complexity(complexity)
                
                formatted_docstring = self.docstring_processor.format(docstring_data)
                
                # Track token usage
                self.token_manager.track_request(usage['prompt_tokens'], usage['completion_tokens'])
                logger.info(f"Tokens used: {usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion")
                
                return ProcessingResult(
                    content=formatted_docstring,
                    usage=usage or {},
                    processing_time=0.0
                )
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_function = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of what the code does."
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the functionality."
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The name of the argument."},
                                    "type": {"type": "string", "description": "The type of the argument."},
                                    "description": {"type": "string", "description": "A description of the argument."}
                                },
                                "required": ["name", "type", "description"]
                            },
                            "description": "A list of arguments, each with a name, type, and description."
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "The return type."},
                                "description": {"type": "string", "description": "A description of the return value."}
                            },
                            "required": ["type", "description"],
                            "description": "An object describing the return value, including its type and a description."
                        },
                        "raises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "exception": {"type": "string", "description": "The type of exception raised."},
                                    "description": {"type": "string", "description": "A description of when the exception is raised."}
                                },
                                "required": ["exception", "description"]
                            },
                            "description": "A list of exceptions that may be raised, each with a type and description."
                        }
                    },
                    "required": ["summary", "description", "args", "returns", "raises"]
                }
            }
        }

    def _create_function_calling_prompt(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> str:
        """
        Create the initial prompt for function calling.

        Args:
            source_code (str): The source code to document.
            metadata (Dict[str, Any]): Additional metadata for the prompt.
            node (Optional[ast.AST]): The AST node representing the code element.

        Returns:
            str: The generated prompt for function calling.
        """
        extraction_data = self.code_extractor.extract_code(source_code)
        complexity_info = self._format_complexity_info(extraction_data)
        dependencies_info = self._format_dependencies_info(extraction_data)
        type_info = self._format_type_info(extraction_data)
        
        prompt = (
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
            "3. CODE CONTEXT:\n"
            f"{complexity_info}\n"
            f"{dependencies_info}\n"
            f"{type_info}\n\n"
            "4. SPECIFIC REQUIREMENTS:\n"
            "- Use exact parameter names and types from the code\n"
            "- Include all type hints in documentation\n"
            "- Document class inheritance where applicable\n"
            "- Note async/generator functions appropriately\n"
            "- Include property decorators in documentation\n\n"
            "The code to document is:\n"
            "```python\n"
            f"{source_code}\n"
            "```\n"
        )
        return prompt

    def _format_complexity_info(self, extraction_data: Any) -> str:
        """
        Format complexity information for the prompt.

        Args:
            extraction_data (Any): Data extracted from the code for complexity analysis.

        Returns:
            str: Formatted complexity information.
        """
        complexity_lines = ["Complexity Information:"]
        for cls in extraction_data.classes:
            score = cls.metrics.get('complexity', 0)
            complexity_lines.append(f"- Class '{cls.name}' complexity: {score}")
            for method in cls.methods:
                m_score = method.metrics.get('complexity', 0)
                complexity_lines.append(f"  - Method '{method.name}' complexity: {m_score}")
        for func in extraction_data.functions:
            score = func.metrics.get('complexity', 0)
            complexity_lines.append(f"- Function '{func.name}' complexity: {score}")
        return "\n".join(complexity_lines)

    def _format_dependencies_info(self, extraction_data: Any) -> str:
        """
        Format dependency information for the prompt.

        Args:
            extraction_data (Any): Data extracted from the code for dependency analysis.

        Returns:
            str: Formatted dependency information.
        """
        dep_lines = ["Dependencies:"]
        for name, deps in extraction_data.imports.items():
            dep_lines.append(f"- {name}: {', '.join(deps)}")
        return "\n".join(dep_lines)

    def _format_type_info(self, extraction_data: Any) -> str:
        """
        Format type information for the prompt.

        Args:
            extraction_data (Any): Data extracted from the code for type analysis.

        Returns:
            str: Formatted type information.
        """
        type_lines = ["Type Information:"]
        for cls in extraction_data.classes:
            type_lines.append(f"Class '{cls.name}':")
            for attr in cls.attributes:
                type_lines.append(f"- {attr['name']}: {attr['type']}")
            for method in cls.methods:
                args_info = [f"{arg.name}: {arg.type_hint}" for arg in method.args]
                type_lines.append(f"- Method '{method.name}({', '.join(args_info)}) -> {method.return_type}'")
        for func in extraction_data.functions:
            args_info = [f"{arg.name}: {arg.type_hint}" for arg in func.args]
            type_lines.append(f"Function '{func.name}({', '.join(args_info)}) -> {func.return_type}'")
        return "\n".join(type_lines)

    def _create_refinement_prompt(self, original_prompt: str, error_message: str, previous_response: dict) -> str:
        """
        Create a refinement prompt, handling previous responses and errors.

        Args:
            original_prompt (str): The original prompt used for function calling.
            error_message (str): Error message to include in the refinement prompt.
            previous_response (dict): Previous response data to include in the prompt.

        Returns:
            str: The refined prompt for further attempts.
        """
        formatted_response = json.dumps(previous_response, indent=4) if previous_response else ""
        prompt = (
            f"{error_message}\n\n"
            + "Previous Response (if any):\n"
            + f"```json\n{formatted_response}\n```\n\n"
            + original_prompt
        )
        return prompt

    def _insert_docstring(self, node: ast.AST, docstring: str) -> None:
        """
        Insert or update docstring in AST node.

        Args:
            node (ast.AST): The AST node to update.
            docstring (str): The docstring to insert.
        """
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return

        docstring_node = ast.Expr(value=ast.Constant(value=docstring))

        if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
            node.body.pop(0)

        node.body.insert(0, docstring_node)

    async def _cache_result(self, cache_key: str, code: str, documentation: str) -> None:
        """
        Cache the processing result.

        Args:
            cache_key (str): The key to use for caching.
            code (str): The processed code.
            documentation (str): The generated documentation.
        """
        try:
            if self.cache:
                await self.cache.save_docstring(
                    cache_key,
                    {'code': code, 'docs': documentation}
                )
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")

    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            if hasattr(self, 'client'):
                await self.client.close()
            if self.cache:
                await self.cache.close()
            if self.metrics_collector:
                await self.metrics_collector.close()
            if self.token_manager:
                await self.token_manager.close()
        except Exception as e:
            logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self):
        """
        Async context manager entry.

        Returns:
            AIInteractionHandler: The instance of the handler.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()