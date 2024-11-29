"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API, handling token management,
caching, and response processing for documentation generation.
"""

from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import json
from dataclasses import dataclass
import ast
from core.logger import LoggerSetup, log_info
from core.cache import Cache
from core.monitoring import MetricsCollector
from core.config import AzureOpenAIConfig
from core.docstring_processor import (
    DocstringProcessor,
    DocstringData,
    DocstringMetrics
)
from api.token_management import TokenManager
from api.api_client import APIClient
from exceptions import ValidationError, ProcessingError, CacheError
from docs.markdown_generator import MarkdownGenerator, MarkdownConfig

logger = LoggerSetup.get_logger(__name__)
config = AzureOpenAIConfig.from_env()

@dataclass
class ProcessingResult:
    """Result of AI processing operation."""
    content: str
    usage: Dict[str, Any]
    metrics: Optional[DocstringMetrics] = None
    cached: bool = False
    processing_time: float = 0.0

class AIInteractionHandler:
    """
    Handles AI interactions for documentation generation via Azure OpenAI API.

    Manages token limits, caching mechanisms, and metrics collection for robust processing.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        token_manager: Optional[TokenManager] = None,
        config: Optional[AzureOpenAIConfig] = None
    ):
        """Initialize the AI Interaction Handler."""
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.metrics = metrics_collector
            self.token_manager = token_manager or TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_name,
                config=self.config
            )
            self.client = APIClient()
            self.docstring_processor = DocstringProcessor()
            self.markdown_generator = MarkdownGenerator(MarkdownConfig())
            self._initialize_tools()
            logger.info("AI Interaction Handler initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Interaction Handler: {str(e)}")
            raise

    def _initialize_tools(self) -> None:
        """Initialize the function tools for structured output."""
        self.docstring_tool = {
            "type": "function",
            "function": {
                "name": "generate_docstring",
                "description": "Generate a Python docstring with structured information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A concise summary of what the code does"
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of the functionality"
                        },
                        "args": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        },
                        "returns": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        },
                        "raises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "description": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["summary", "description"]
                }
            }
        }

    async def _generate_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    def _create_documentation_prompt(self, source_code: str, metadata: Dict[str, Any], node: Optional[ast.AST] = None) -> str:
        """Create a dynamic prompt for documentation generation."""
        prompt = [
            "You are a highly skilled Python documentation expert. Your task is to generate comprehensive documentation for the following Python code.",
            "Please analyze the code carefully and provide detailed documentation using the generate_docstring function.",
            "",
            "Follow these requirements:",
            "1. Provide a clear, concise summary of the code's purpose",
            "2. Write a detailed description explaining how it works",
            "3. Document all parameters with accurate types and descriptions",
            "4. Include return value documentation if applicable",
            "5. List any exceptions that may be raised",
            "6. Keep descriptions practical and implementation-focused",
            "",
            "The documentation should follow Python's Google-style format.",
            "",
            "Here's the code to document:",
            "```python",
            source_code,
            "```",
            "",
            "Respond using the generate_docstring function to provide the documentation."
        ]
        
        return "\n".join(prompt)

    def _format_docstring(self, data: Dict[str, Any]) -> str:
        """Format the docstring data into a string."""
        docstring_lines = []
        
        # Add summary
        if data.get("summary"):
            docstring_lines.append(data["summary"])
            docstring_lines.append("")
        
        # Add description
        if data.get("description"):
            docstring_lines.append(data["description"])
            docstring_lines.append("")
        
        # Add arguments
        if data.get("args"):
            docstring_lines.append("Args:")
            for arg in data["args"]:
                docstring_lines.append(f"    {arg['name']} ({arg.get('type', 'Any')}): {arg.get('description', '')}")
            docstring_lines.append("")
        
        # Add returns
        if data.get("returns"):
            docstring_lines.append("Returns:")
            docstring_lines.append(f"    {data['returns'].get('type', 'Any')}: {data['returns'].get('description', '')}")
            docstring_lines.append("")
        
        # Add raises
        if data.get("raises"):
            docstring_lines.append("Raises:")
            for exc in data["raises"]:
                docstring_lines.append(f"    {exc.get('type', 'Exception')}: {exc.get('description', '')}")
        
        return "\n".join(docstring_lines).strip()
    
    async def _check_cache(self, cache_key: str) -> Optional[Tuple[str, str]]:
        """
        Check if result is available in cache.

        Args:
            cache_key: The cache key to check

        Returns:
            Optional[Tuple[str, str]]: Cached code and documentation if available
        """
        try:
            if not self.cache:
                return None
                
            logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = await self.cache.get_cached_docstring(cache_key)
            
            if cached_data and isinstance(cached_data, dict):
                code = cached_data.get('code')
                docs = cached_data.get('docs')
                if code and docs:
                    logger.info(f"Cache hit for key: {cache_key}")
                    return code, docs
                
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            raise CacheError(f"Failed to check cache: {str(e)}")

    async def _cache_result(
        self,
        cache_key: str,
        code: str,
        documentation: str
    ) -> None:
        """
        Cache the processing result.

        Args:
            cache_key: The cache key
            code: The processed code
            documentation: The generated documentation
        """
        try:
            if not self.cache:
                return
                
            await self.cache.save_docstring(
                cache_key,
                {
                    'code': code,
                    'docs': documentation
                }
            )
            logger.debug(f"Cached result for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to cache result: {str(e)}")
            logger.warning("Continuing without caching")
            
    async def process_code(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self._generate_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise
        
    def cleanup(self) -> None:
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                # Remove read-only attribute from Git files
                for root, dirs, files in os.walk(self.temp_dir):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o777)
                
                shutil.rmtree(self.temp_dir, onerror=self._handle_remove_error)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up directory: {e}")

    def _handle_remove_error(self, func, path, exc_info):
        """
        Error handler for shutil.rmtree.
        
        Args:
            func: The function that failed
            path: The path being processed
            exc_info: Exception information
        """
        try:
            os.chmod(path, 0o777)
            func(path)
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")
            
    async def close(self):
        """Close the AI interaction handler."""
        if self.client:
            await self.client.close()

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise

    async def generate_markdown_documentation(
        self,
        source_code: str,
        metadata: Dict[str, Any],
        node: Optional[ast.AST] = None
    ) -> Optional[ProcessingResult]:
        """Generate markdown documentation using Azure OpenAI with function calling."""
        try:
            prompt = self._create_documentation_prompt(source_code, metadata, node)
            start_time = datetime.now()

            response, usage = await self.client.process_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Increase max tokens for longer documentation
                tools=[self.docstring_tool],
                tool_choice={"type": "function", "function": {"name": "generate_docstring"}}
            )

            if not response:
                logger.error("No response received from API")
                return None

            processing_time = (datetime.now() - start_time).total_seconds()

            # Extract message content from various response formats
            message_content = None
            if isinstance(response, dict):
                # Direct message format
                message_content = response.get("content")
                
                # Check in message object
                if not message_content and "message" in response:
                    message_content = response["message"].get("content")
                
                # Check in choices array
                if not message_content and "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice:
                            message_content = choice["message"].get("content")
                        else:
                            message_content = choice.get("content")

            # Create docstring data structure
            docstring_data = {
                "summary": message_content[:100] if message_content else "Generated documentation",
                "description": message_content if message_content else source_code,
                "args": [],
                "returns": {"type": "Any", "description": "Documentation generation result"}
            }

            # Try to extract function call data if available
            function_data = None
            if "tool_calls" in response:
                tool_calls = response.get("tool_calls", [])
                if tool_calls:
                    function_data = tool_calls[0].get("function", {})
            elif "function_call" in response:
                function_data = {
                    'name': response["function_call"].get("name"),
                    'arguments': response["function_call"].get("arguments")
                }
            elif "message" in response and "function_call" in response.get("message", {}):
                function_data = response["message"]["function_call"]
            elif "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice and "function_call" in choice["message"]:
                    function_data = choice["message"]["function_call"]

            # Process function data if available
            if function_data and function_data.get("arguments"):
                try:
                    if isinstance(function_data["arguments"], str):
                        parsed_args = json.loads(function_data["arguments"])
                    else:
                        parsed_args = function_data["arguments"]
                    docstring_data.update(parsed_args)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse function arguments: {e}")

            # Return processed result
            return ProcessingResult(
                content=self._format_docstring(docstring_data),
                usage=usage or {},
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            raise ProcessingError(f"Failed to generate documentation: {str(e)}")

    async def process_code_with_markdown(self, source_code: str, cache_key: Optional[str] = None) -> Tuple[str, str]:
        """Process source code to generate markdown documentation."""
        try:
            if not source_code or not source_code.strip():
                raise ValidationError("Empty source code provided")

            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self._check_cache(cache_key)
                    if cached_result:
                        return cached_result
                except CacheError as e:
                    logger.warning(f"Cache error, proceeding without cache: {str(e)}")

            # Generate documentation
            result = await self.generate_markdown_documentation(source_code, {})
            if not result or not result.content:
                raise ProcessingError("Documentation generation failed")

            # Update code with documentation
            updated_code = f'"""\n{result.content}\n"""\n\n{source_code}'

            # Cache result if enabled
            if self.cache and cache_key:
                await self._cache_result(cache_key, updated_code, result.content)

            return updated_code, result.content

        except Exception as e:
            logger.error(f"Process code failed: {str(e)}")
            raise
