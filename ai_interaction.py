"""AI Interaction Handler Module.

Manages interactions with Azure OpenAI API, focusing on prompt generation
and API communication with validated data.
"""

from typing import Any, Dict, Optional
from openai import AsyncAzureOpenAI

from core.cache import Cache
from core.config import AzureOpenAIConfig
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.response_parsing import ResponseParsingService
from api.token_management import TokenManager
from core.types import DocumentationData, DocstringData
from exceptions import ValidationError, DocumentationError
from core.utils import FileUtils
from core.docstring_processor import DocstringProcessor

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    """Handler for AI interactions with Azure OpenAI API.
    
    Focuses on dynamic prompt generation and processing AI responses.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None,
    ) -> None:
        """Initialize the AIInteractionHandler."""
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache  # Cache instance injected
            self.token_manager = token_manager or TokenManager()
            self.response_parser = response_parser or ResponseParsingService()
            self.metrics = metrics
            self.docstring_processor = DocstringProcessor()  # You'll need to import this

            
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    def _generate_cache_key(self, extracted_info: Dict[str, Any]) -> str:
        """Generate a unique cache key based on input data."""
        try:
            # Use the utility function to generate a cache key
            return FileUtils.generate_cache_key(str(extracted_info))
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return None

    async def process_code(
        self,
        extracted_info: Dict[str, Any],
        cache_key: Optional[str] = None,
    ) -> Optional[DocumentationData]:
        """Process code information and generate documentation.

        Args:
            extracted_info: Pre-processed code information from extractors
            cache_key: Optional cache key for storing results

        Returns:
            Optional[DocumentationData]: AI-generated documentation data
        """
        try:
            # Generate cache key if not provided
            cache_key = cache_key or self._generate_cache_key(extracted_info)
            
            # Check cache if enabled
            if self.cache and cache_key:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        return DocumentationData(**cached_result)
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Generate prompt
            prompt = self._generate_prompt(extracted_info)

            # Get request parameters from token manager
            request_params = await self.token_manager.validate_and_prepare_request(prompt)

            # Make API request
            completion = await self.client.chat.completions.create(**request_params)

            # Process completion through token manager
            content, usage = await self.token_manager.process_completion(completion)

            if not content:
                raise ValidationError("Empty response from AI service")

            # Parse and validate
            parsed_response = await self.response_parser.parse_response(
                response=content,
                expected_format="docstring",
                validate_schema=True,
            )

            if not parsed_response.validation_success:
                self.logger.warning(f"Response validation had errors: {parsed_response.errors}")
                return None

            # Create documentation data
            doc_data = DocumentationData(
                module_info={
                    "name": extracted_info.get("module_name", "Unknown"),
                    "file_path": extracted_info.get("file_path", "Unknown")
                },
                ai_content=parsed_response.content,
                docstring_data=self.docstring_processor.parse(parsed_response.content),
                code_metadata=extracted_info,
                source_code=extracted_info.get("source_code"),
                metrics=extracted_info.get("metrics")
            )

            # Cache the result if enabled
            if self.cache and cache_key:
                try:
                    await self.cache.save_docstring(
                        cache_key,
                        doc_data.to_dict(),  # Convert to dictionary for caching
                        expire=self.config.cache_ttl
                    )
                    self.logger.debug(f"Cached result for key: {cache_key}")
                except Exception as e:
                    self.logger.error(f"Cache storage error: {e}")

            return doc_data

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None

    def _generate_prompt(self, extracted_info: Dict[str, Any]) -> str:
        """Generate a dynamic prompt based on extracted code information.

        Args:
            extracted_info: Extracted code information and metadata

        Returns:
            str: Generated prompt for AI service
        """
        context_blocks = []

        # Add code context
        if "source" in extracted_info:
            context_blocks.append(
                "CODE TO DOCUMENT:\n"
                "```python\n"
                f"{extracted_info['source']}\n"
                "```\n"
            )

        # Add existing docstring context if available
        if "existing_docstring" in extracted_info:
            context_blocks.append(
                "EXISTING DOCUMENTATION:\n"
                f"{extracted_info['existing_docstring']}\n"
            )

        # Add code complexity information
        if "metrics" in extracted_info:
            metrics = extracted_info["metrics"]
            context_blocks.append(
                "CODE METRICS:\n"
                f"- Complexity: {metrics.get('complexity', 'N/A')}\n"
                f"- Maintainability: {metrics.get('maintainability_index', 'N/A')}\n"
            )

        # Add function/method context
        if "args" in extracted_info:
            args_info = "\n".join(
                f"- {arg['name']}: {arg['type']}"
                for arg in extracted_info["args"]
            )
            context_blocks.append(
                "FUNCTION ARGUMENTS:\n"
                f"{args_info}\n"
            )

        # Add return type information
        if "returns" in extracted_info:
            context_blocks.append(
                "RETURN TYPE:\n"
                f"{extracted_info['returns']['type']}\n"
            )

        # Add exception information
        if "raises" in extracted_info:
            raises_info = "\n".join(
                f"- {exc['exception']}"
                for exc in extracted_info["raises"]
            )
            context_blocks.append(
                "EXCEPTIONS RAISED:\n"
                f"{raises_info}\n"
            )

        # Combine all context blocks
        context = "\n\n".join(context_blocks)

        # Add the base prompt template
        prompt_template = (
            "Generate documentation for the provided code as a JSON object.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "```json\n"
            "{\n"
            '  "summary": "A brief one-line summary of the function/method",\n'
            '  "description": "Detailed description of the functionality",\n'
            '  "args": [\n'
            "    {\n"
            '      "name": "string - parameter name",\n'
            '      "type": "string - parameter data type",\n'
            '      "description": "string - brief description of the parameter"\n'
            "    }\n"
            "  ],\n"
            '  "returns": {\n'
            '    "type": "string - return data type",\n'
            '    "description": "string - brief description of return value"\n'
            "  },\n"
            '  "raises": [\n'
            "    {\n"
            '      "exception": "string - exception class name",\n'
            '      "description": "string - circumstances under which raised"\n'
            "    }\n"
            "  ],\n"
            '  "complexity": "integer - McCabe complexity score"\n'
            "}\n"
            "```\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "IMPORTANT NOTES:\n"
            "1. Always include a 'complexity' field with an integer value\n"
            "2. If complexity > 10, note this in the description with [WARNING]\n"
            "3. Never set complexity to null or omit it\n"
            "4. Provide detailed, specific descriptions\n"
            "5. Ensure all type hints are accurate Python types\n\n"
            "Respond with only the JSON object. Do not include any other text."
        )

        return prompt_template

    async def invalidate_cache(self, extracted_info: Dict[str, Any]) -> bool:
        """Invalidate cache for specific code information."""
        try:
            if not self.cache:
                return True

            cache_key = self._generate_cache_key(extracted_info)
            if cache_key:
                await self.cache.invalidate(cache_key)
                self.logger.info(f"Invalidated cache for key: {cache_key}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if not self.cache:
                return {"enabled": False}
            return await self.cache.get_stats()
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def close(self) -> None:
        """Close and clean up resources."""
        try:
            if self.client:
                await self.client.close()
            if self.cache:
                await self.cache.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        await self.close()