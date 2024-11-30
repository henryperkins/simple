# ai_interaction.py

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


@dataclass
class DocstringData:
    """Data structure for holding docstring information."""
    summary: str
    description: str
    args: list
    returns: dict
    raises: list
    complexity: int


DOCSTRING_SCHEMA = {
    "name": "google_style_docstring",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A brief summary of the method or function."
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the method or function."
            },
            "args": {
                "type": "array",
                "description": "A list of arguments for the method or function.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the argument."
                        },
                        "type": {
                            "type": "string",
                            "description": "The data type of the argument."
                        },
                        "description": {
                            "type": "string",
                            "description": "A brief description of the argument."
                        }
                    },
                    "required": [
                        "name",
                        "type",
                        "description"
                    ],
                    "additionalProperties": false
                }
            },
            "returns": {
                "type": "object",
                "description": "Details about the return value of the method or function.",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The data type of the return value."
                    },
                    "description": {
                        "type": "string",
                        "description": "A brief description of the return value."
                    }
                },
                "required": [
                    "type",
                    "description"
                ],
                "additionalProperties": false
            },
            "raises": {
                "type": "array",
                "description": "A list of exceptions that may be raised by the method or function.",
                "items": {
                    "type": "object",
                    "properties": {
                        "exception": {
                            "type": "string",
                            "description": "The name of the exception that may be raised."
                        },
                        "description": {
                            "type": "string",
                            "description": "A brief description of the circumstances under which the exception is raised."
                        }
                    },
                    "required": [
                        "exception",
                        "description"
                    ],
                    "additionalProperties": false
                }
            },
            "complexity": {
                "type": "integer",
                "description": "McCabe complexity score"
            }
        },
        "required": [
            "summary",
            "description",
            "args",
            "returns",
            "raises",
            "complexity"
        ],
        "additionalProperties": false
    }
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
                config=self.config,
                metrics_collector=self.metrics_collector
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

                if node and self.metrics_collector:
                    complexity = self.metrics_collector.calculate_complexity(node)
                    response_data["complexity"] = complexity  # Assign complexity directly

                validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])  # Correct schema used here
                docstring_data = DocstringData(**response_data)
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
                        },
                        "complexity": {
                            "type": "integer",
                            "description": "McCabe complexity score"
                        }
                    },
                    "required": ["summary", "description", "args", "returns", "raises", "complexity"]
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

        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body[0] = docstring_node
        else:
            node.body.insert(0, docstring_node)

    async def _cache_result(self, cache_key: str, updated_code: str, module_docs: str) -> None:
        """Cache the result of the code processing."""
        if not self.cache:
            return
        try:
            await self.cache.store(cache_key, {"updated_code": updated_code, "module_docs": module_docs})
            logger.info(f"Cached result for key: {cache_key}")
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
            # Removed the call to close on TokenManager
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
