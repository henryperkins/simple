"""AI service module for interacting with OpenAI API."""
import json
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
import asyncio
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path

from core.logger import LoggerSetup
from core.config import AIConfig
from core.cache import Cache
from core.exceptions import ProcessingError, ConnectionError
from core.types.extraction_types import ExtractionResult
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.types.base import (
    DocumentationContext, 
    ProcessingResult, 
    ExtractedClass, 
    ExtractedFunction,
    DocstringData,
    DocumentationData
)
from api.token_management import TokenManager

class AIService:
    """Service for interacting with OpenAI API."""

    def __init__(self, config: AIConfig, correlation_id: Optional[str] = None) -> None:
        """Initialize AI service.
        
        Args:
            config: AI service configuration
            correlation_id: Optional correlation ID for tracking related operations
        """
        self.config = config
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(__name__)
        self.cache = Cache()
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
        self._client = None
        self.docstring_processor = DocstringProcessor()
        self.response_parser = ResponseParsingService(correlation_id)
        self.token_manager = TokenManager(
            model=self.config.model,
            config=self.config
        )
        # Define the function schema for structured output
        self.function_schema = {
            "name": "generate_docstring",
            "description": "Generate Google-style documentation for code",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief one-line summary of what the code does"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of the functionality and purpose"
                    },
                    "args": {
                        "type": "array",
                        "description": "List of arguments for the method or function",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the argument"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "The data type of the argument"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A brief description of the argument"
                                }
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "description": "Details about the return value",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "The data type of the return value"
                            },
                            "description": {
                                "type": "string",
                                "description": "A brief description of the return value"
                            }
                        },
                        "required": ["type", "description"]
                    },
                    "raises": {
                        "type": "array",
                        "description": "List of exceptions that may be raised",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exception": {
                                    "type": "string",
                                    "description": "The name of the exception that may be raised"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "A brief description of when this exception is raised"
                                }
                            },
                            "required": ["exception", "description"]
                        }
                    },
                    "complexity": {
                        "type": "integer",
                        "description": "McCabe complexity score"
                    }
                },
                "required": ["summary", "description", "args", "returns", "raises", "complexity"]
            }
        }

    def _format_function_info(self, func: ExtractedFunction) -> str:
        """Format function information for prompt.
        
        Args:
            func: The extracted function information
            
        Returns:
            Formatted function string
        """
        args_str = ", ".join(f"{arg.name}: {arg.type or 'Any'}" for arg in func.args)
        return (
            f"Function: {func.name}\n"
            f"Arguments: ({args_str})\n"
            f"Returns: {func.returns.get('type', 'Any')}\n"
            f"Existing Docstring: {func.docstring if func.docstring else 'None'}\n"
            f"Decorators: {', '.join(func.decorators) if func.decorators else 'None'}\n"
            f"Is Async: {'Yes' if func.is_async else 'No'}\n"
            f"Complexity Score: {func.metrics.cyclomatic_complexity if func.metrics else 'Unknown'}\n"
        )

    def _format_class_info(self, cls: ExtractedClass) -> str:
        """Format class information for prompt.
        
        Args:
            cls: The extracted class information
            
        Returns:
            Formatted class string
        """
        methods_str = "\n    ".join(
            f"- {m.name}({', '.join(a.name for a in m.args)})" 
            for m in cls.methods
        )
        return (
            f"Class: {cls.name}\n"
            f"Base Classes: {', '.join(cls.bases) if cls.bases else 'None'}\n"
            f"Existing Docstring: {cls.docstring if cls.docstring else 'None'}\n"
            f"Methods:\n    {methods_str}\n"
            f"Attributes: {', '.join(a['name'] for a in cls.attributes)}\n"
            f"Instance Attributes: {', '.join(a['name'] for a in cls.instance_attributes)}\n"
            f"Decorators: {', '.join(cls.decorators) if cls.decorators else 'None'}\n"
            f"Is Exception: {'Yes' if cls.is_exception else 'No'}\n"
            f"Complexity Score: {cls.metrics.cyclomatic_complexity if cls.metrics else 'Unknown'}\n"
        )

    async def enhance_and_format_docstring(self, context: DocumentationContext) -> ProcessingResult:
        """Enhance and format docstrings using AI.
        
        Args:
            context: Documentation context containing source code and metadata
            
        Returns:
            ProcessingResult containing enhanced documentation
            
        Raises:
            ProcessingError: If enhancement fails
        """
        try:
            # Create cache key based on source code and metadata
            cache_key = context.get_cache_key()
            cached = self.cache.get(cache_key)
            if cached:
                # Validate cached content through docstring processor
                docstring_data = self.docstring_processor.parse(cached)
                is_valid, validation_errors = self.docstring_processor.validate(docstring_data)
                
                if is_valid:
                    return ProcessingResult(
                        content=cached,
                        usage={},
                        metrics={},
                        is_cached=True,
                        processing_time=0.0,
                        validation_status=True,
                        validation_errors=[],
                        schema_errors=[]
                    )
                else:
                    self.logger.warning(f"Cached content failed validation: {validation_errors}")
                    # Remove invalid entry from cache dictionary
                    del self.cache.cache[cache_key]

            # Extract relevant information from context
            module_name = context.metadata.get("module_name", "")
            file_path = context.metadata.get("file_path", "")
            
            # Build comprehensive prompt with detailed code structure
            prompt = (
                f"Generate comprehensive Google-style documentation for the following Python module.\n\n"
                f"Module Name: {module_name}\n"
                f"File Path: {file_path}\n\n"
                "Code Structure:\n\n"
            )

            # Add class information
            if context.classes:
                prompt += "Classes:\n"
                for cls in context.classes:
                    prompt += self._format_class_info(cls)
                prompt += "\n"

            # Add function information
            if context.functions:
                prompt += "Functions:\n"
                for func in context.functions:
                    prompt += self._format_function_info(func)
                prompt += "\n"

            # Add source code
            prompt += (
                "Source Code:\n"
                f"{context.source_code}\n\n"
                "Analyze the code and generate comprehensive Google-style documentation. "
                "Include a brief summary, detailed description, arguments, return values, and possible exceptions. "
                "Ensure all descriptions are clear and technically accurate."
            )

            # Get AI response using function calling
            start_time = datetime.now()
            response = await self._make_api_call(prompt)
            
            # Extract the function call response
            if "choices" in response and response["choices"]:
                message = response["choices"][0]["message"]
                if "function_call" in message:
                    function_args = json.loads(message["function_call"]["arguments"])
                    parsed_response = await self.response_parser.parse_response(
                        function_args,
                        expected_format="docstring"
                    )
                else:
                    raise ProcessingError("No function call in response")
            else:
                raise ProcessingError("Invalid response format")

            if not parsed_response.validation_success:
                raise ProcessingError("Failed to validate AI response")

            # Process through docstring processor for additional validation
            docstring_data = self.docstring_processor.parse(parsed_response.content)
            is_valid, validation_errors = self.docstring_processor.validate(docstring_data)

            if not is_valid:
                self.logger.warning(f"Generated docstring failed validation: {validation_errors}")
                # Try to fix common issues
                fixed_content = self._fix_common_docstring_issues(parsed_response.content)
                docstring_data = self.docstring_processor.parse(fixed_content)
                is_valid, validation_errors = self.docstring_processor.validate(docstring_data)
                
                if not is_valid:
                    raise ProcessingError(f"Failed to generate valid docstring: {validation_errors}")
                parsed_response.content = fixed_content

            processing_time = (datetime.now() - start_time).total_seconds()

            # Create DocumentationData
            doc_data = DocumentationData(
                module_name=module_name,
                module_path=Path(file_path),
                module_summary=docstring_data.summary,
                source_code=context.source_code,
                docstring_data=docstring_data,
                ai_content=parsed_response.content,
                code_metadata={
                    "classes": [cls.to_dict() for cls in (context.classes or [])],
                    "functions": [func.to_dict() for func in (context.functions or [])],
                    "constants": context.constants or []
                },
                validation_status=is_valid,
                validation_errors=validation_errors
            )

            # Create ProcessingResult
            result = ProcessingResult(
                content=doc_data.to_dict(),
                usage=response.get("usage", {}),
                metrics={
                    "processing_time": processing_time,
                    "response_size": len(str(response)),
                    "validation_success": is_valid
                },
                is_cached=False,
                processing_time=processing_time,
                validation_status=is_valid,
                validation_errors=validation_errors,
                schema_errors=[]
            )

            # Only cache if validation passed
            if is_valid:
                self.cache.set(cache_key, parsed_response.content)

            return result

        except Exception as e:
            self.logger.error(f"Error enhancing docstring: {str(e)}", exc_info=True)
            raise ProcessingError(f"Failed to enhance docstring: {str(e)}")

    def _fix_common_docstring_issues(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common docstring validation issues.
        
        Args:
            content: The docstring content to fix
            
        Returns:
            Fixed docstring content
        """
        fixed = content.copy()
        
        # Ensure required fields exist
        if "summary" not in fixed:
            fixed["summary"] = ""
        if "description" not in fixed:
            fixed["description"] = fixed.get("summary", "")
        if "args" not in fixed:
            fixed["args"] = []
        if "returns" not in fixed:
            fixed["returns"] = {"type": "None", "description": ""}
        if "raises" not in fixed:
            fixed["raises"] = []
        if "complexity" not in fixed:
            fixed["complexity"] = 1

        # Ensure args have required fields
        for arg in fixed["args"]:
            if "name" not in arg:
                arg["name"] = "unknown"
            if "type" not in arg:
                arg["type"] = "Any"
            if "description" not in arg:
                arg["description"] = ""

        # Ensure returns has required fields
        if isinstance(fixed["returns"], dict):
            if "type" not in fixed["returns"]:
                fixed["returns"]["type"] = "None"
            if "description" not in fixed["returns"]:
                fixed["returns"]["description"] = ""
        else:
            fixed["returns"] = {"type": "None", "description": ""}

        # Ensure raises have required fields
        for exc in fixed["raises"]:
            if "exception" not in exc:
                exc["exception"] = "Exception"
            if "description" not in exc:
                exc["description"] = ""

        return fixed

    async def generate_documentation(self, code: str, context: Dict[str, Any] = None) -> str:
        """Generate documentation for code using AI.
        
        Args:
            code: Source code to generate documentation for
            context: Optional additional context for generation
            
        Returns:
            Generated documentation string
        """
        cache_key = f"doc_{hash(code)}_{hash(str(context))}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        prompt = self._build_prompt(code, context)
        
        try:
            async with self.semaphore:
                response = await self._make_api_call(prompt)
                documentation = self._parse_response(response)
                
            self.cache.set(cache_key, documentation)
            return documentation
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {str(e)}")
            raise

    def _build_prompt(self, code: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for AI model.
        
        Args:
            code: Source code to document
            context: Optional additional context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Generate Google-style documentation for the following code:\n\n{code}\n"
        
        if context:
            prompt += f"\nAdditional context:\n{json.dumps(context, indent=2)}"
            
        return prompt

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """Make API call to OpenAI.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            API response dictionary
            
        Raises:
            Exception: If API call fails
        """
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json"
        }
        
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        # Add function calling parameters
        request_params["functions"] = [self.function_schema]
        request_params["function_call"] = {"name": "generate_docstring"}

        try:
            async with aiohttp.ClientSession() as session:
                # Ensure endpoint ends with a slash for proper URL joining
                endpoint = self.config.endpoint.rstrip('/') + '/'
                # Construct the URL path
                path = f"openai/deployments/{self.config.deployment}/chat/completions"
                # Join the URL properly
                url = urljoin(endpoint, path) + "?api-version=2024-02-15-preview"
                
                async with session.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=self.config.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API call failed with status {response.status}: {error_text}")
                        
                    response_data = await response.json()
                    content, usage = await self.token_manager.process_completion(response_data)
                    return response_data
                    
        except asyncio.TimeoutError:
            raise Exception(f"API call timed out after {self.config.timeout} seconds")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """Parse API response to extract documentation.
        
        Args:
            response: Raw API response dictionary
            
        Returns:
            Extracted documentation string
            
        Raises:
            Exception: If response parsing fails
        """
        try:
            if "choices" in response and response["choices"]:
                message = response["choices"][0]["message"]
                if "function_call" in message:
                    return message["function_call"]["arguments"]
                else:
                    return message["content"].strip()
            raise Exception("Invalid response format")
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")

    async def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality using AI.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing quality metrics and suggestions
        """
        prompt = f"Analyze the following code for quality and provide specific improvements:\n\n{code}"
        
        try:
            async with self.semaphore:
                response = await self._make_api_call(prompt)
                
            analysis = self._parse_response(response)
            return {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "correlation_id": self.correlation_id
            }
            
        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {str(e)}")
            raise

    async def batch_process(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple items concurrently.
        
        Args:
            items: List of items to process, each containing 'code' and optional 'context'
            
        Returns:
            List of results corresponding to input items
        """
        tasks = []
        for item in items:
            if "code" not in item:
                raise ValueError("Each item must contain 'code' key")
                
            task = self.generate_documentation(
                item["code"],
                item.get("context")
            )
            tasks.append(task)
            
        try:
            results = await asyncio.gather(*tasks)
            return [{"documentation": result} for result in results]
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise

    async def test_connection(self) -> None:
        """Test the connection to the AI service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.endpoint,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    headers={"api-key": self.config.api_key}
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise ConnectionError(f"Connection failed: {response_text}")
            self.logger.info("Connection test successful", extra={'correlation_id': self.correlation_id})
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})
            raise

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._client:
                await self._client.close()
            if self.cache:
                await self.cache.close()
            self.logger.info("AI service cleanup completed", extra={'correlation_id': self.correlation_id})
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True, extra={'correlation_id': self.correlation_id})

    async def __aenter__(self) -> "AIService":
        """Async context manager entry."""
        await self.test_connection()
        return self

    async def __aexit__(self, exc_type: Optional[BaseException], exc_val: Optional[BaseException], exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
