# gemini_model.py
from typing import Dict, Any, Optional, List, Tuple
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import asyncio
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from base_client import BaseAIClient
from core.config import GeminiConfig
from core.logger import log_info, log_error, log_debug

class GeminiClient(BaseAIClient):
    """
    Gemini AI client implementation with advanced features including:
    - Structured output generation
    - Document processing
    - Error handling and retries
    - Authentication management
    """

    def __init__(self, config: GeminiConfig):
        """Initialize Gemini client with configuration."""
        self.config = config
        
        # Configure Gemini
        genai.configure(api_key=config.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=config.model_name,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            }
        )
        
        log_info(f"Initialized Gemini client with model: {config.model_name}")

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
        Generate docstring using Gemini with advanced error handling and retries.
        """
        try:
            # Create prompt
            prompt = self._create_prompt(
                func_name=func_name,
                params=params,
                return_type=return_type,
                complexity_score=complexity_score,
                existing_docstring=existing_docstring,
                decorators=decorators,
                exceptions=exceptions
            )

            # Generate response with structured output
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                    "candidate_count": 1
                }
            )

            # Parse and validate response
            if response and response.text:
                parsed_response = self._parse_response(response)
                if self._validate_response(parsed_response):
                    log_info(f"Successfully generated docstring for {func_name}")
                    return parsed_response
                else:
                    log_error(f"Invalid response format for {func_name}")
                    return None

            log_error(f"Empty response from Gemini for {func_name}")
            return None

        except Exception as e:
            log_error(f"Error generating docstring with Gemini: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of Gemini service.
        """
        try:
            start_time = time.time()
            response = await self.model.generate_content_async(
                "Health check",
                generation_config={"max_output_tokens": 10}
            )
            latency = time.time() - start_time

            return {
                "status": "healthy" if response else "unhealthy",
                "latency": round(latency, 3),
                "model": self.config.model_name,
                "error": None
            }

        except Exception as e:
            log_error(f"Gemini health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "latency": None,
                "model": self.config.model_name,
                "error": str(e)
            }

    def _create_prompt(self, **kwargs) -> str:
        """
        Create a detailed prompt for Gemini.
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

    def _parse_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse Gemini's response into structured format."""
        try:
            # Extract JSON from response text
            text = response.text.strip()
            # Find JSON content between curly braces
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                content = json.loads(json_str)
                
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": response.prompt_token_count,
                        "completion_tokens": response.completion_token_count,
                        "total_tokens": response.total_token_count
                    }
                }
            return None

        except json.JSONDecodeError as e:
            log_error(f"Error decoding JSON response: {str(e)}")
            return None
        except Exception as e:
            log_error(f"Error parsing Gemini response: {str(e)}")
            return None

    def _validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate Gemini's response format and content.
        """
        required_fields = ["docstring", "summary", "complexity_analysis", "examples"]
        
        try:
            if not response or "content" not in response:
                return False

            content = response["content"]
            return all(field in content for field in required_fields)

        except Exception as e:
            log_error(f"Error validating Gemini response: {str(e)}")
            return False

    async def batch_process(
        self,
        prompts: List[str],
        batch_size: int = 5,
        delay: float = 0.1
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Process multiple prompts in batches.
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
        
        
[Timestamp: 2024/11/17 09:16:59]

**user**: Continue 
[Timestamp: 2024/11/17 09:17:03]

**ai**: Here's the continuation of the Gemini client implementation:

```python
# gemini_client.py (continued)

    async def _process_single_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Process a single prompt with error handling."""
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens
                }
            )
            return self._parse_response(response)
        except Exception as e:
            log_error(f"Error processing prompt: {str(e)}")
            return None

    async def stream_completion(
        self,
        prompt: str,
        callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Stream completion responses with optional callback.

        Args:
            prompt: The input prompt
            callback: Optional callback function for processing stream chunks

        Returns:
            Optional[str]: Complete generated text or None if failed
        """
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens
                },
                stream=True
            )

            collected_chunks = []
            async for chunk in response:
                if chunk.text:
                    collected_chunks.append(chunk.text)
                    if callback:
                        await callback(chunk.text)

            return ''.join(collected_chunks)

        except Exception as e:
            log_error(f"Error in stream completion: {str(e)}")
            return None

    async def process_document(
        self,
        document: str,
        extraction_type: str = "general"
    ) -> Optional[Dict[str, Any]]:
        """
        Process document content with structured data extraction.

        Args:
            document: Document content to process
            extraction_type: Type of extraction to perform

        Returns:
            Optional[Dict[str, Any]]: Extracted structured data or None if failed
        """
        try:
            prompt = self._create_document_prompt(document, extraction_type)
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more focused extraction
                    "max_output_tokens": self.config.max_tokens
                }
            )
            
            return self._parse_structured_response(response)

        except Exception as e:
            log_error(f"Error processing document: {str(e)}")
            return None

    def _create_document_prompt(self, document: str, extraction_type: str) -> str:
        """Create prompt for document processing."""
        return f"""
Extract structured information from the following document.
Extraction type: {extraction_type}

Document content:
{document}

Please provide a JSON response with extracted information following this structure:
{{
    "main_topics": ["topic1", "topic2"],
    "key_points": ["point1", "point2"],
    "entities": {{"name": "type"}},
    "relationships": ["relationship1", "relationship2"],
    "metadata": {{
        "document_type": "type",
        "confidence_score": 0.95
    }}
}}
"""

    def _parse_structured_response(self, response) -> Optional[Dict[str, Any]]:
        """Parse structured response from document processing."""
        try:
            if not response or not response.text:
                return None

            # Extract JSON content
            text = response.text.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
            if json_match:
                content = json.loads(json_match.group())
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": response.prompt_token_count,
                        "completion_tokens": response.completion_token_count,
                        "total_tokens": response.total_token_count
                    }
                }
            return None

        except Exception as e:
            log_error(f"Error parsing structured response: {str(e)}")
            return None

    async def close(self):
        """Close the Gemini client connection."""
        try:
            # Clean up any resources
            self.model = None
            log_info("Gemini client closed successfully")
        except Exception as e:
            log_error(f"Error closing Gemini client: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
