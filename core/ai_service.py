"""AI service module for interacting with OpenAI API."""

import json
from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path

from core.logger import LoggerSetup
from core.config import AIConfig
from core.cache import Cache
from core.exceptions import ProcessingError
from core.docstring_processor import DocstringProcessor
from core.response_parsing import ResponseParsingService
from core.prompt_manager import PromptManager
from core.types.base import DocumentationContext, ProcessingResult, DocumentationData
from api.token_management import TokenManager


class AIService:
    """Service for interacting with OpenAI API."""

    def __init__(
        self,
        config: AIConfig,
        correlation_id: Optional[str] = None,
        docstring_processor: DocstringProcessor = None,
        response_parser: ResponseParsingService = None,
        token_manager: TokenManager = None,
        prompt_manager: Optional[PromptManager] = None
    ) -> None:
        """Initialize AI service with dependency injection.

        Args:
            config: AI service configuration
            correlation_id: Optional correlation ID for tracking related operations
            docstring_processor: Docstring processor instance
            response_parser: Response parsing service instance
            token_manager: Token manager instance
        """
        self.config = config
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(__name__)
        self.cache = Cache()
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls
        self._client = None

        # Inject dependencies
        self.docstring_processor = docstring_processor or DocstringProcessor()
        self.response_parser = response_parser or ResponseParsingService(correlation_id)
        self.token_manager = token_manager or TokenManager(model=self.config.model, config=self.config)
        self.prompt_manager = prompt_manager or PromptManager(correlation_id)


    async def enhance_and_format_docstring(
        self, context: DocumentationContext
    ) -> ProcessingResult:
        """Enhance and format docstrings using AI.

        Args:
            context: Documentation context containing source code and metadata

        Returns:
            ProcessingResult containing enhanced documentation

        Raises:
            ProcessingError: If enhancement fails
        """
        try:
            self.logger.info("Starting AI response processing...")
            # Create cache key based on source code and metadata
            cache_key = context.get_cache_key()
            cached = self.cache.get(cache_key)
            if cached:
                # Validate cached content through docstring processor
                docstring_data = self.docstring_processor.parse(cached)
                is_valid, validation_errors = self.docstring_processor.validate(
                    docstring_data
                )

                if is_valid:
                    self.logger.info("AI response processing completed. Cached content used.")
                    return ProcessingResult(
                        content=cached,
                        usage={},
                        metrics={},
                        is_cached=True,
                        processing_time=0.0,
                        validation_status=True,
                        validation_errors=[],
                        schema_errors=[],
                    )
                else:
                    self.logger.warning(
                        f"Cached content failed validation: {validation_errors}"
                    )
                    # Remove invalid entry from cache dictionary
                    del self.cache.cache[cache_key]

            # Extract relevant information from context
            module_name = context.metadata.get("module_name", "")
            file_path = context.metadata.get("file_path", "")

            prompt = self.prompt_manager.create_documentation_prompt(
                module_name=module_name,
                file_path=file_path,
                source_code=context.source_code,
                classes=context.classes,
                functions=context.functions
            )

            # Get AI response using function calling with chunking if needed
            start_time = datetime.now()
            if len(prompt) > self.config.max_tokens // 2:
                # Split into chunks and process separately
                chunks = self._split_prompt(prompt)
                responses = []
                for chunk in chunks:
                    chunk_response = await self._make_api_call(chunk)
                    responses.append(chunk_response)
                response = self._merge_responses(responses)
            else:
                response = await self._make_api_call(prompt)

            # Extract the function call response
            if "choices" in response and response["choices"]:
                message = response["choices"][0]["message"]
                if "function_call" in message:
                    function_args = json.loads(message["function_call"]["arguments"])
                    parsed_response = await self.response_parser.parse_response(
                        function_args, expected_format="docstring"
                    )
                else:
                    raise ProcessingError("No function call in response")
            else:
                raise ProcessingError("Invalid response format")

            if not parsed_response.validation_success:
                raise ProcessingError("Failed to validate AI response")

            # Process through docstring processor for additional validation
            docstring_data = self.docstring_processor.parse(parsed_response.content)
            is_valid, validation_errors = self.docstring_processor.validate(
                docstring_data
            )

            if not is_valid:
                self.logger.warning(
                    f"Generated docstring failed validation: {validation_errors}"
                )
                # Try to fix common issues
                fixed_content = self._fix_common_docstring_issues(
                    parsed_response.content
                )
                docstring_data = self.docstring_processor.parse(fixed_content)
                is_valid, validation_errors = self.docstring_processor.validate(
                    docstring_data
                )

                if not is_valid:
                    raise ProcessingError(
                        f"Failed to generate valid docstring: {validation_errors}"
                    )
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
                    "constants": context.constants or [],
                },
                validation_status=is_valid,
                validation_errors=validation_errors,
            )

            # Create ProcessingResult
            result = ProcessingResult(
                content=doc_data.to_dict(),
                usage=response.get("usage", {}),
                metrics={
                    "processing_time": processing_time,
                    "response_size": len(str(response)),
                    "validation_success": is_valid,
                },
                is_cached=False,
                processing_time=processing_time,
                validation_status=is_valid,
                validation_errors=validation_errors,
                schema_errors=[],
            )

            # Only cache if validation passed
            if is_valid:
                self.cache.set(cache_key, parsed_response.content)

            self.logger.info(f"AI response processing completed. Processed {len(context.functions) + len(context.classes)} items.")
            return result

        except Exception as e:
            self.logger.error(f"Error enhancing docstring: {str(e)}", exc_info=True)
            raise ProcessingError(f"Failed to enhance docstring: {str(e)}") from e

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

    async def generate_documentation(
        self, code: str, context: Dict[str, Any] = None
    ) -> str:
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
        return self.prompt_manager.create_documentation_prompt(
            module_name=context.get("module_name", "") if context else "",
            file_path=context.get("file_path", "") if context else "",
            source_code=code,
            classes=context.get("classes", []) if context else [],
            functions=context.get("functions", []) if context else []
        )

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """Make API call to OpenAI.

        Args:
            prompt: The prompt to send to the API

        Returns:
            API response dictionary

        Raises:
            Exception: If API call fails
        """
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}

        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Add function calling parameters
        request_params["functions"] = [self.prompt_manager.get_function_schema()]
        request_params["function_call"] = {"name": "generate_docstring"}

        try:
            async with aiohttp.ClientSession() as session:
                # Ensure endpoint ends with a slash for proper URL joining
                endpoint = self.config.endpoint.rstrip("/") + "/"
                # Construct the URL path
                path = f"openai/deployments/{self.config.deployment}/chat/completions"
                # Join the URL properly
                url = urljoin(endpoint, path) + "?api-version=2024-02-15-preview"

                self.logger.debug(f"Making API call to {url} with prompt: {prompt[:100]}...")

                async with session.post(
                    url,
                    headers=headers,
                    json=request_params,
                    timeout=self.config.timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(
                            f"API call failed with status {response.status}: {error_text}"
                        )
                        raise ProcessingError(
                            f"API call failed with status {response.status}: {error_text}"
                        ) from aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_text
                        )

                    response_data = await response.json()
                    self.logger.debug(f"API response received: {str(response_data)[:100]}...")
                    content, usage = await self.token_manager.process_completion(
                        response_data
                    )
                    return response_data

        except asyncio.TimeoutError:
            self.logger.error(f"API call timed out after {self.config.timeout} seconds")
            raise Exception(f"API call timed out after {self.config.timeout} seconds")
        except Exception as e:
            self.logger.error(f"API call failed: {str(e)}", exc_info=True)
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
            self.logger.error(f"Failed to parse API response: {str(e)}", exc_info=True)
            raise Exception(f"Failed to parse API response: {str(e)}")

    async def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality using AI.

        Args:
            code: Source code to analyze

        Returns:
            Dictionary containing quality metrics and suggestions
        """
        prompt = self.prompt_manager.create_code_analysis_prompt(code)

        try:
            async with self.semaphore:
                response = await self._make_api_call(prompt)

            analysis = self._parse_response(response)
            return {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "correlation_id": self.correlation_id,
            }

        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {str(e)}", exc_info=True)
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
                self.logger.error("Each item must contain 'code' key")
                raise ValueError("Each item must contain 'code' key")

            task = self.generate_documentation(item["code"], item.get("context"))
            tasks.append(task)

        try:
            results = await asyncio.gather(*tasks)
            return [{"documentation": result} for result in results]
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
            raise

    async def test_connection(self) -> None:
        """Test the connection to the AI service."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.endpoint,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    headers={"api-key": self.config.api_key},
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        self.logger.error(f"Connection failed: {response_text}")
                        raise ConnectionError(f"Connection failed: {response_text}")
            self.logger.info(
                "Connection test successful",
                extra={"correlation_id": self.correlation_id},
            )
        except Exception as e:
            self.logger.error(
                f"Connection test failed: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )
            raise

    def _split_prompt(self, prompt: str) -> List[str]:
        """Split a large prompt into smaller chunks."""
        chunk_size = self.config.max_tokens // 2
        words = prompt.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # Add 1 for space
            if current_size + word_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _merge_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple responses into a single response."""
        if not responses:
            return {}
            
        merged = responses[0].copy()
        for response in responses[1:]:
            if "choices" in response and response["choices"]:
                message = response["choices"][0]["message"]
                if "function_call" in message:
                    args = json.loads(message["function_call"]["arguments"])
                    merged_args = json.loads(merged["choices"][0]["message"]["function_call"]["arguments"])
                    
                    # Merge descriptions
                    if "description" in args:
                        merged_args["description"] = merged_args.get("description", "") + "\n" + args["description"]
                    
                    # Merge other fields
                    for key in ["args", "raises"]:
                        if key in args:
                            merged_args[key].extend(args[key])
                            
                    merged["choices"][0]["message"]["function_call"]["arguments"] = json.dumps(merged_args)
                    
            if "usage" in response:
                for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    merged["usage"][key] = merged["usage"].get(key, 0) + response["usage"].get(key, 0)
                    
        return merged

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._client:
                await self._client.close()
            if self.cache:
                await self.cache.close()
            self.logger.info(
                "AI service cleanup completed",
                extra={"correlation_id": self.correlation_id},
            )
        except Exception as e:
            self.logger.error(
                f"Error during cleanup: {e}",
                exc_info=True,
                extra={"correlation_id": self.correlation_id},
            )

    async def __aenter__(self) -> "AIService":
        """Async context manager entry."""
        await self.test_connection()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
