# ai_interaction.py

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
from core.code_extraction import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.types import ProcessingResult, DocstringData, AIHandler, DocumentationContext
from api.token_management import TokenManager
from api.api_client import APIClient
from exceptions import ValidationError, ExtractionError

logger = LoggerSetup.get_logger(__name__)

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
                    "additionalProperties": False
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
                "additionalProperties": False
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
                    "additionalProperties": False
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
        "additionalProperties": False
    }
}

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
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_calculator: Optional[Metrics] = None,
        token_manager: Optional[TokenManager] = None,
        config: Optional[AzureOpenAIConfig] = None,
        code_extractor: Optional[CodeExtractor] = None
    ) -> None:
        """
        Initialize the AI Interaction Handler.

        Args:
            cache (Optional[Cache]): Optional cache for storing intermediate results.
            metrics_calculator (Optional[Metrics]): Optional metrics calculator for code analysis.
            token_manager (Optional[TokenManager]): Optional token manager for handling token limits.
            config (Optional[AzureOpenAIConfig]): Configuration settings for Azure OpenAI service.
            code_extractor (Optional[CodeExtractor]): Extractor for analyzing and extracting code elements.

        Raises:
            Exception: If initialization fails.
        """
        self.logger = LoggerSetup.get_logger(__name__)

        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.metrics_calculator = metrics_calculator or Metrics()
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
            self.logger.info("AI Interaction Handler initialized successfully")

            openai.api_type = "azure"
            openai.api_key = self.config.api_key
            openai.api_base = self.config.endpoint
            openai.api_version = self.config.api_version

            self._current_module_tree = None
            self._current_module_docs = {}
            self._current_module = None

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise
        
    def _preprocess_code(self, source_code: str) -> str:
        """
        Preprocess source code before parsing to handle special cases.
        
        Args:
            source_code (str): The source code to preprocess.
            
        Returns:
            str: Preprocessed source code.
        """
        try:
            # Handle timestamps in comments by converting them to strings
            import re
            
            # Pattern to match timestamps (various formats)
            timestamp_pattern = r'(\[|\b)(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)\]?'
            
            def timestamp_replacer(match):
                # Convert timestamp to string format
                timestamp = match.group(2)
                # Keep the opening bracket if it exists
                prefix = match.group(1) if match.group(1) == '[' else ''
                # Add closing bracket if we had an opening one
                suffix = ']' if prefix == '[' else ''
                return f'{prefix}"{timestamp}"{suffix}'

            processed_code = re.sub(timestamp_pattern, timestamp_replacer, source_code)
            
            self.logger.debug("Preprocessed source code to handle timestamps")
            return processed_code

        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Process source code to generate documentation.

        Args:
            source_code (str): The source code to process.
            cache_key (Optional[str]): Optional cache key for storing results.
            extracted_info (Optional[Dict[str, Any]]): Optional pre-extracted code information.

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or None if processing fails.

        Raises:
            ExtractionError: If there is an error in extracting code.
            ValidationError: If there is an error in validating the response.
        """
        try:
            # Check cache first if key provided
            if cache_key and self.cache:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        return cached_result.get("updated_code"), cached_result.get("documentation")
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Preprocess code to handle timestamps and other special cases
            processed_code = self._preprocess_code(source_code)

            # Parse the processed code
            try:
                tree = ast.parse(processed_code)
            except SyntaxError as e:
                self.logger.error(f"Syntax error in source code: {e}")
                raise ExtractionError(f"Failed to parse code: {e}")

            # Use provided metadata or extract it
            if extracted_info:
                metadata = extracted_info
            else:
                self.logger.debug("Processing code with extractor...")
                extraction_result = self.code_extractor.extract_code(processed_code)
                if not extraction_result:
                    self.logger.error("Failed to process code with extractor")
                    return None
                    
                # Create metadata dictionary from extraction result
                metadata = {
                    'module_docstring': extraction_result.module_docstring,
                    'classes': [
                        {
                            'name': cls.name,
                            'docstring': cls.docstring,
                            'methods': [
                                {
                                    'name': method.name,
                                    'docstring': method.docstring,
                                    'args': [
                                        {'name': arg.name, 'type': arg.type_hint}
                                        for arg in method.args
                                    ],
                                    'return_type': method.return_type,
                                    'complexity': method.metrics.get('complexity', 0) if method.metrics else 0
                                }
                                for method in cls.methods
                            ],
                            'complexity': cls.metrics.get('complexity', 0) if cls.metrics else 0
                        }
                        for cls in extraction_result.classes
                    ],
                    'functions': [
                        {
                            'name': func.name,
                            'docstring': func.docstring,
                            'args': [
                                {'name': arg.name, 'type': arg.type_hint}
                                for arg in func.args
                            ],
                            'return_type': func.return_type,
                            'complexity': func.metrics.get('complexity', 0) if func.metrics else 0
                        }
                        for func in extraction_result.functions
                    ],
                    'metrics': extraction_result.metrics or {}
                }

            # Create the prompt
            prompt = self._create_function_calling_prompt(processed_code, metadata)
            
            # Get response from AI
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

            # Process the response
            message = response.choices[0].message
            if not message.tool_calls or not message.tool_calls[0].function:
                self.logger.error("No function call in response")
                return None

            # Debug: Log the raw function arguments
            function_args = message.tool_calls[0].function.arguments
            self.logger.debug(f"Raw function arguments received from AI: {function_args}")

            # Parse function arguments with resilient error handling
            try:
                response_data = json.loads(function_args)
                # Debug: Log the parsed response data
                self.logger.debug(f"Parsed response data: {json.dumps(response_data, indent=2)}")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error - Raw content that failed: {function_args}")
                self.logger.warning(f"Failed to parse AI-generated documentation: {e}")
                # Create a basic valid response instead of failing
                response_data = {
                    "summary": "Generated documentation",
                    "description": "Documentation could not be fully parsed",
                    "args": [],
                    "returns": {"type": "Any", "description": "Unknown return value"},
                    "raises": [],
                    "complexity": 1
                }

            # Validate response data with detailed error reporting
            try:
                validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
            except ValidationError as e:
                self.logger.error(f"Validation error - Schema path: {' -> '.join(str(p) for p in e.path)}")
                self.logger.error(f"Validation error message: {e.message}")
                self.logger.error(f"Failed validating value: {e.instance}")
                response_data = self._create_fallback_response(response_data)

            # Create docstring data
            try:
                docstring_data = DocstringData(**response_data)
                self.logger.debug(f"Created DocstringData object successfully: {docstring_data}")
            except Exception as e:
                self.logger.error(f"Error creating DocstringData: {e}")
                return None
            
            # Format the docstring
            docstring = self.docstring_processor.format(docstring_data)
            self.logger.debug(f"Formatted docstring result: {docstring}")

            # Update source code with new docstring
            try:
                tree = ast.parse(processed_code)
                modified = False
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Get existing docstring
                        existing_docstring = ast.get_docstring(node) or ""
                        updated_docstring = self.docstring_processor.update_docstring(
                            existing_docstring, docstring
                        )
                        self.docstring_processor.insert_docstring(node, updated_docstring)
                        modified = True

                updated_code = ast.unparse(tree) if modified else processed_code
            except Exception as e:
                self.logger.error(f"Error updating source code: {e}")
                return None

            # Track usage if available
            if usage:
                self.token_manager.track_request(
                    usage.get('prompt_tokens', 0),
                    usage.get('completion_tokens', 0)
                )

            # Cache result if caching is enabled
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
        
        # Try to salvage any valid fields from the partial data
        for key in fallback.keys():
            if key in partial_data and isinstance(partial_data[key], type(fallback[key])):
                fallback[key] = partial_data[key]
                
        return fallback
    async def _generate_documentation(self, source_code: str, metadata: Dict[str, Any], node: ast.AST) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI."""
        try:
            prompt = self._create_function_calling_prompt(source_code, metadata, node)
            
            # Create function-specific context
            context = {
                'name': getattr(node, 'name', 'unknown'),
                'type': type(node).__name__,
                'args': self._get_function_args(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [],
                'returns': self._get_return_annotation(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                'is_class': isinstance(node, ast.ClassDef),
                'bases': self._get_base_classes(node) if isinstance(node, ast.ClassDef) else [],
                'decorators': self._get_decorators(node),
                'complexity': self.metrics_calculator.calculate_complexity(node) if self.metrics_calculator else 0
            }

            # Add the context to the prompt
            prompt += f"\nContext:\n{json.dumps(context, indent=2)}"
            
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

                # Add complexity score
                if context['complexity'] > 0:
                    response_data["complexity"] = context['complexity']

                validate(instance=response_data, schema=DOCSTRING_SCHEMA["schema"])
                docstring_data = DocstringData(**response_data)
                formatted_docstring = self.docstring_processor.format(docstring_data)

                self.token_manager.track_request(usage['prompt_tokens'], usage['completion_tokens'])
                
                return ProcessingResult(
                    content=formatted_docstring,
                    usage=usage or {},
                    processing_time=0.0
                )

        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            return None
        
    def _get_function_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict[str, Any]]:
        """Extract function arguments with type hints."""
        args = []
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else 'Any',
                'has_default': False
            }
            args.append(arg_info)
        
        # Handle defaults
        defaults = node.args.defaults
        if defaults:
            default_offset = len(args) - len(defaults)
            for i, default in enumerate(defaults):
                args[default_offset + i]['has_default'] = True
                args[default_offset + i]['default'] = ast.unparse(default)
        
        return args

    def _get_return_annotation(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
        """Extract return type annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _get_base_classes(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        return [ast.unparse(base) for base in node.bases]

    def _get_decorators(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> List[str]:
        """Extract decorator names."""
        return [ast.unparse(decorator) for decorator in node.decorator_list]
    
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
        """
        Cache the result of the code processing.

        Args:
            cache_key (str): The cache key for storing results.
            updated_code (str): The updated source code.
            module_docs (str): The generated documentation.

        Raises:
            Exception: If caching fails.
        """
        if not self.cache:
            return
        try:
            await self.cache.store(cache_key, {"updated_code": updated_code, "module_docs": module_docs})
            self.logger.info(f"Cached result for key: {cache_key}")
        except Exception as e:
            self.logger.error(f"Failed to cache result: {e}")

    async def close(self) -> None:
        """
        Close and cleanup resources.

        Raises:
            Exception: If closing resources fails.
        """
        try:
            if hasattr(self, 'client'):
                await self.client.close()
            if self.cache:
                await self.cache.close()
            if hasattr(self, 'metrics_calculator') and self.metrics_calculator:  # Changed from metrics_collector
                await self.metrics_calculator.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """
        Async context manager entry.

        Returns:
            AIInteractionHandler: The instance of the handler.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager exit.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()
