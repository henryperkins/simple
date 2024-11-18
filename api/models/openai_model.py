# openai_client.py
from typing import Dict, Any, Optional, List, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import json
import time
import asyncio
from base_client import BaseAIClient
from core.config import OpenAIConfig
from core.logger import log_info, log_error, log_debug
from core.exceptions import TooManyRetriesError

class OpenAIClient(BaseAIClient):
    """
    OpenAI client implementation with advanced features including:
    - Streaming responses
    - Token counting and optimization
    - Rate limit handling
    - Structured output parsing
    - Comprehensive error handling
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize OpenAI client with configuration."""
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key)
        
        # Initialize token counter
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        log_info(f"Initialized OpenAI client with model: {config.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_docstring(
        self,
        func_name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        complexity_score: int,
        existing_docstring: str,
        decorators: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate docstring using OpenAI with advanced error handling and retries.
        """
        try:
            # Create messages for the conversation
            messages = [
                {
                    "role": "system",
                    "content": "You are a technical documentation expert. Generate comprehensive and accurate function documentation."
                },
                {
                    "role": "user",
                    "content": self._create_prompt(
                        func_name=func_name,
                        params=params,
                        return_type=return_type,
                        complexity_score=complexity_score,
                        existing_docstring=existing_docstring,
                        decorators=decorators,
                        exceptions=exceptions
                    )
                }
            ]

            # Validate token count
            total_tokens = self._count_tokens(json.dumps(messages))
            if total_tokens > self.config.max_tokens * 0.8:  # 80% threshold
                messages = self._optimize_prompt(messages)

            # Request completion with structured output
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"}
            )

            # Parse and validate response
            if response and response.choices:
                parsed_response = self._parse_response(response)
                if self._validate_response(parsed_response):
                    log_info(f"Successfully generated docstring for {func_name}")
                    return parsed_response
                else:
                    log_error(f"Invalid response format for {func_name}")
                    return None

            log_error(f"Empty response from OpenAI for {func_name}")
            return None

        except Exception as e:
            log_error(f"Error generating docstring with OpenAI: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of OpenAI service.
        """
        try:
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=10
            )
            latency = time.time() - start_time

            return {
                "status": "healthy" if response else "unhealthy",
                "latency": round(latency, 3),
                "model": self.config.model_name,
                "error": None
            }

        except Exception as e:
            log_error(f"OpenAI health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "latency": None,
                "model": self.config.model_name,
                "error": str(e)
            }

    def _create_prompt(self, **kwargs) -> str:
        """
        Create a detailed prompt for OpenAI.
        """
        return f"""
Generate documentation for the following Python function:

Function Name: {kwargs['func_name']}
Parameters: {', '.join(f'{name}: {type_}' for name, type_ in kwargs['params'])}
Return Type: {kwargs['return_type']}
Decorators: {', '.join(kwargs['decorators']) if kwargs['decorators'] else 'None'}
Exceptions: {', '.join(kwargs['exceptions']) if kwargs['exceptions'] else 'None'}
Complexity Score: {kwargs['complexity_score']}
Existing Docstring: {kwargs['existing_docstring'] if kwargs['existing_docstring'] else 'None'}

Please provide a JSON response with the following structure:
{{
    "docstring": "The complete docstring text",
    "summary": "A brief summary of the function's purpose",
    "complexity_analysis": "Time and space complexity analysis",
    "examples": ["Usage example 1", "Usage example 2"],
    "changelog": "Documentation changelog"
}}

Ensure the docstring follows Google style format and includes:
1. Clear description of functionality
2. Detailed parameter descriptions with types
3. Return value description
4. Usage examples
5. Time and space complexity analysis
6. Exception documentation
"""

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            log_error(f"Error counting tokens: {str(e)}")
            return 0

    def _optimize_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Optimize prompt to fit within token limits."""
        try:
            # Start with system message
            optimized = [messages[0]]
            current_tokens = self._count_tokens(json.dumps(optimized))
            
            # Add user message content with optimization if needed
            user_content = messages[1]["content"]
            sections = user_content.split('\n\n')
            
            essential_sections = []
            optional_sections = []
            
            # Categorize sections
            for section in sections:
                if any(key in section.lower() for key in ["function name", "parameters", "return type"]):
                    essential_sections.append(section)
                else:
                    optional_sections.append(section)
            
            # Add essential sections
            optimized_content = '\n\n'.join(essential_sections)
            
            # Add optional sections that fit
            for section in optional_sections:
                section_tokens = self._count_tokens(section)
                if current_tokens + section_tokens < self.config.max_tokens * 0.8:
                    optimized_content += f"\n\n{section}"
                    current_tokens += section_tokens
            
            optimized.append({"role": "user", "content": optimized_content})
            return optimized
            
        except Exception as e:
            log_error(f"Error optimizing prompt: {str(e)}")
            return messages

    def _parse_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse OpenAI's response into structured format."""
        try:
            if not response.choices or not response.choices[0].message.content:
[Timestamp: 2024/11/17 09:05:46]

**user**: continue 
[Timestamp: 2024/11/17 09:05:55]

**ai**: Here's the continuation of the OpenAI client implementation:

```python
# openai_client.py (continued)

    def _parse_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse OpenAI's response into structured format."""
        try:
            if not response.choices or not response.choices[0].message.content:
                return None

            # Parse JSON content
            content = json.loads(response.choices[0].message.content)
            
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except json.JSONDecodeError as e:
            log_error(f"Error decoding JSON response: {str(e)}")
            return None
        except Exception as e:
            log_error(f"Error parsing OpenAI response: {str(e)}")
            return None

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate OpenAI's response format and content.
        """
        required_fields = ["docstring", "summary", "complexity_analysis", "examples"]
        
        try:
            if not response or "content" not in response:
                return False

            content = response["content"]
            return all(field in content for field in required_fields)

        except Exception as e:
            log_error(f"Error validating OpenAI response: {str(e)}")
            return False

    async def stream_completion(
        self,
        prompt: str,
        callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Stream completion responses with optional callback for real-time processing.

        Args:
            prompt: The input prompt
            callback: Optional callback function for processing stream chunks

        Returns:
            Optional[str]: Complete generated text or None if failed
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            collected_chunks = []
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    collected_chunks.append(chunk.choices[0].delta.content)
                    if callback:
                        await callback(chunk.choices[0].delta.content)

            return ''.join(collected_chunks)

        except Exception as e:
            log_error(f"Error in stream completion: {str(e)}")
            return None

    async def batch_process(
        self,
        prompts: List[str],
        batch_size: int = 5,
        delay: float = 0.1
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process multiple prompts in batches with rate limiting.

        Args:
            prompts: List of prompts to process
            batch_size: Number of concurrent requests
            delay: Delay between batches in seconds

        Returns:
            List[Optional[Dict[str, Any]]]: List of responses
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_tasks = []
            
            for prompt in batch:
                task = asyncio.create_task(self._process_single_prompt(prompt))
                batch_tasks.append(task)
            
            # Process batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Rate limiting delay
            if i + batch_size < len(prompts):
                await asyncio.sleep(delay)
        
        return results

    async def _process_single_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Process a single prompt with error handling."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return self._parse_response(response)
        except Exception as e:
            log_error(f"Error processing prompt: {str(e)}")
            return None

    async def close(self):
        """Close the OpenAI client connection."""
        try:
            if hasattr(self.client, 'close'):
                await self.client.close()
            log_info("OpenAI client closed successfully")
        except Exception as e:
            log_error(f"Error closing OpenAI client: {str(e)}")

    def get_token_count(self, text: str) -> int:
        """
        Get token count for text using model-specific tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            int: Number of tokens
        """
        return self._count_tokens(text)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
