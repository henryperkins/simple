import aiohttp
import asyncio
import json
import os
import re
from typing import Any, Dict, Optional, Union, List, Iterable, cast, TypedDict, Literal
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from core.logger import LoggerSetup
from utils import validate_schema
from openai import AzureOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from anthropic import Anthropic
from anthropic.types import Message, MessageParam

# Initialize logger
logger = LoggerSetup.get_logger("api_interaction")

# Load environment variables
load_dotenv()

class ParameterProperty(TypedDict):
    type: str
    description: str

class Parameters(TypedDict):
    type: Literal["object"]
    properties: Dict[str, ParameterProperty]
    required: List[str]

class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: Parameters

def extract_section(text: str, section_name: str) -> str:
    """Extract a section from Claude's response."""
    pattern = rf"{section_name}:\s*(.*?)(?=\n\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_parameter_section(text: str) -> List[Dict[str, str]]:
    """Extract parameter information from Claude's response."""
    params_section = extract_section(text, "Parameters")
    params = []
    param_pattern = r"(\w+)\s*\(([^)]+)\):\s*(.+?)(?=\n\w+\s*\(|\Z)"
    for match in re.finditer(param_pattern, params_section, re.DOTALL):
        params.append({
            "name": match.group(1),
            "type": match.group(2).strip(),
            "description": match.group(3).strip()
        })
    return params

def extract_return_section(text: str) -> Dict[str, str]:
    """Extract return information from Claude's response."""
    returns_section = extract_section(text, "Returns")
    type_pattern = r"(\w+):\s*(.+)"
    match = re.search(type_pattern, returns_section)
    return {
        "type": match.group(1) if match else "None",
        "description": match.group(2).strip() if match else ""
    }

def extract_code_examples(text: str) -> List[str]:
    """Extract code examples from Claude's response."""
    examples_section = extract_section(text, "Examples")
    examples = []
    for match in re.finditer(r"```python\s*(.*?)\s*```", examples_section, re.DOTALL):
        examples.append(match.group(1).strip())
    return examples

def format_response(sections: Dict[str, Any]) -> Dict[str, Any]:
    """Format parsed sections into a standardized response."""
    return {
        "summary": sections.get("summary", "No summary available"),
        "docstring": sections.get("summary", "No documentation available"),
        "params": sections.get("params", []),
        "returns": sections.get("returns", {"type": "None", "description": ""}),
        "examples": sections.get("examples", []),
        "classes": sections.get("classes", [])  # Ensure classes is included
    }

class APIClient:
    """Unified API client for multiple LLM providers."""
    
    def __init__(self):
        # Initialize API clients
        self.azure_client = self._init_azure_client()
        self.openai_client = self._init_openai_client()
        self.anthropic_client = self._init_anthropic_client()
        
        # Configuration
        self.azure_deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
        self.openai_model = "gpt-4-turbo-preview"  # Updated to latest model
        self.claude_model = "claude-3-opus-20240229"

    def _init_azure_client(self) -> Optional[AzureOpenAI]:
        """Initialize Azure OpenAI client."""
        try:
            if os.getenv("AZURE_OPENAI_API_KEY"):
                return AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_ENDPOINT", "https://api.azure.com"),
                    api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
                    azure_deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4"),
                    azure_ad_token=os.getenv("AZURE_AD_TOKEN"),
                    azure_ad_token_provider=None  # Add token provider if needed
                )
            return None
        except Exception as e:
            logger.error(f"Error initializing Azure client: {e}")
            return None

    def _init_openai_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client."""
        try:
            if os.getenv("OPENAI_API_KEY"):
                return OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_API_BASE"),  # Updated from api_base
                    timeout=60.0,
                    max_retries=3
                )
            return None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return None

    def _init_anthropic_client(self) -> Optional[Anthropic]:
        """Initialize Anthropic client."""
        try:
            if os.getenv("ANTHROPIC_API_KEY"):
                return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            return None
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            return None

class ClaudeResponseParser:
    """Handles Claude-specific response parsing and formatting."""
    
    @staticmethod
    def parse_function_analysis(response: str) -> Dict[str, Any]:
        """Parse Claude's natural language response into structured format."""
        try:
            # Extract key sections from Claude's response
            sections = {
                'summary': extract_section(response, 'Summary'),
                'params': extract_parameter_section(response),
                'returns': extract_return_section(response),
                'examples': extract_code_examples(response),
                'classes': []  # Ensure classes is included
            }
            
            # Validate and format response
            return format_response(sections)
            
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            return ClaudeResponseParser.get_default_response()

    @staticmethod
    def get_default_response() -> Dict[str, Any]:
        """Return a default response in case of parsing errors."""
        return {
            "summary": "Error parsing response",
            "docstring": "Error occurred while parsing the documentation.",
            "params": [],
            "returns": {"type": "None", "description": ""},
            "examples": [],
            "classes": []  # Ensure classes is included
        }

class DocumentationAnalyzer:
    """Handles code analysis and documentation generation."""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.function_schema = self._get_function_schema()

    def _get_function_schema(self) -> FunctionDefinition:
        """Get the function schema for documentation generation."""
        return {
            "name": "generate_documentation",
            "description": "Generates documentation for code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the code"
                    },
                    "docstring": {
                        "type": "string",
                        "description": "Detailed documentation"
                    },
                    "params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Parameter name"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Parameter type"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Parameter description"
                                }
                            },
                            "required": ["name", "type", "description"]
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "Return type"
                            },
                            "description": {
                                "type": "string",
                                "description": "Return value description"
                            }
                        },
                        "required": ["type", "description"]
                    },
                    "examples": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Code example"
                        }
                    },
                    "classes": {  # Ensure classes is included
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Class name"
                                },
                                "docstring": {
                                    "type": "string",
                                    "description": "Class documentation"
                                },
                                "methods": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Method name"
                                            },
                                            "docstring": {
                                                "type": "string",
                                                "description": "Method documentation"
                                            },
                                            "params": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": {
                                                            "type": "string",
                                                            "description": "Parameter name"
                                                        },
                                                        "type": {
                                                            "type": "string",
                                                            "description": "Parameter type"
                                                        },
                                                        "has_type_hint": {
                                                            "type": "boolean",
                                                            "description": "Whether the parameter has a type hint"
                                                        }
                                                    },
                                                    "required": ["name", "type", "has_type_hint"]
                                                }
                                            },
                                            "returns": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "description": "Return type"
                                                    },
                                                    "has_type_hint": {
                                                        "type": "boolean",
                                                        "description": "Whether the return type has a type hint"
                                                    }
                                                },
                                                "required": ["type", "has_type_hint"]
                                            },
                                            "complexity_score": {
                                                "type": "integer",
                                                "description": "Complexity score of the method"
                                            },
                                            "line_number": {
                                                "type": "integer",
                                                "description": "Line number where the method starts"
                                            },
                                            "end_line_number": {
                                                "type": "integer",
                                                "description": "Line number where the method ends"
                                            },
                                            "code": {
                                                "type": "string",
                                                "description": "Code of the method"
                                            },
                                            "is_async": {
                                                "type": "boolean",
                                                "description": "Whether the method is asynchronous"
                                            },
                                            "is_generator": {
                                                "type": "boolean",
                                                "description": "Whether the method is a generator"
                                            },
                                            "is_recursive": {
                                                "type": "boolean",
                                                "description": "Whether the method is recursive"
                                            },
                                            "summary": {
                                                "type": "string",
                                                "description": "Summary of the method"
                                            },
                                            "changelog": {
                                                "type": "string",
                                                "description": "Changelog of the method"
                                            }
                                        },
                                        "required": ["name", "docstring", "params", "returns", "complexity_score", "line_number", "end_line_number", "code", "is_async", "is_generator", "is_recursive", "summary", "changelog"]
                                    }
                                },
                                "attributes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Attribute name"
                                            },
                                            "type": {
                                                "type": "string",
                                                "description": "Attribute type"
                                            },
                                            "line_number": {
                                                "type": "integer",
                                                "description": "Line number where the attribute is defined"
                                            }
                                        },
                                        "required": ["name", "type", "line_number"]
                                    }
                                },
                                "instance_variables": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "Instance variable name"
                                            },
                                            "line_number": {
                                                "type": "integer",
                                                "description": "Line number where the instance variable is defined"
                                            }
                                        },
                                        "required": ["name", "line_number"]
                                    }
                                },
                                "base_classes": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "Base class name"
                                    }
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Summary of the class"
                                },
                                "changelog": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "change": {
                                                "type": "string",
                                                "description": "Description of the change"
                                            },
                                            "timestamp": {
                                                "type": "string",
                                                "description": "Timestamp of the change"
                                            }
                                        }
                                    }
                                }
                            },
                            "required": ["name", "docstring", "methods", "attributes", "instance_variables", "base_classes", "summary", "changelog"]
                        }
                    }
                },
                "required": ["summary", "docstring", "params", "returns", "classes"]
            }
        }

    async def make_api_request(
        self,
        messages: List[Dict[str, str]],
        service: str,
        temperature: float = 0.1
    ) -> Any:
        """Make API request to specified service."""
        logger.info(f"Making API request to {service}")
        
        retries = 3
        base_backoff = 2

        for attempt in range(retries):
            try:
                if service == "azure" and self.api_client.azure_client:
                    return await self._azure_request(messages, temperature)
                elif service == "openai" and self.api_client.openai_client:
                    return await self._openai_request(messages, temperature)
                elif service == "claude" and self.api_client.anthropic_client:
                    return await self._claude_request(messages, temperature)
                else:
                    raise ValueError(f"Invalid service: {service}")

            except Exception as e:
                logger.error(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(base_backoff ** attempt)
                else:
                    raise

    async def _azure_request(self, messages: List[Dict[str, str]], temperature: float) -> ChatCompletion:
        """Make request to Azure OpenAI."""
        if not self.api_client.azure_client:
            raise ValueError("Azure client is not initialized")
            
        chat_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=msg["content"]) if msg["role"] == "system"
            else ChatCompletionUserMessageParam(role="user", content=msg["content"])
            for msg in messages
        ]
            
        tools: List[ChatCompletionToolParam] = [{
            "type": "function",
            "function": cast(FunctionDefinition, self.function_schema)
        }]
            
        return await asyncio.to_thread(
            self.api_client.azure_client.chat.completions.create,
            model=self.api_client.azure_deployment,
            messages=chat_messages,
            temperature=temperature,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_documentation"}}
        )

    async def _openai_request(self, messages: List[Dict[str, str]], temperature: float) -> ChatCompletion:
        """Make request to OpenAI."""
        if not self.api_client.openai_client:
            raise ValueError("OpenAI client is not initialized")
            
        chat_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=msg["content"]) if msg["role"] == "system"
            else ChatCompletionUserMessageParam(role="user", content=msg["content"])
            for msg in messages
        ]
            
        tools: List[ChatCompletionToolParam] = [{
            "type": "function",
            "function": cast(FunctionDefinition, self.function_schema)
        }]
            
        return await asyncio.to_thread(
            self.api_client.openai_client.chat.completions.create,
            model=self.api_client.openai_model,
            messages=chat_messages,
            temperature=temperature,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_documentation"}},
            response_format={"type": "json_object"}
        )

    async def _claude_request(self, messages: List[Dict[str, str]], temperature: float) -> Message:
        """Make request to Anthropic Claude."""
        if not self.api_client.anthropic_client:
            raise ValueError("Anthropic client is not initialized")
            
        system_message = "You are an expert code documentation generator."
        
        claude_messages: List[MessageParam] = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": cast(Literal["user", "assistant"], msg["role"]),
                    "content": msg["content"]
                })
            
        return await asyncio.to_thread(
            self.api_client.anthropic_client.messages.create,
            model=self.api_client.claude_model,
            messages=claude_messages,
            temperature=temperature,
            max_tokens=2000,
            system=system_message
        )

async def analyze_function_with_openai(
    function_details: Dict[str, Any],
    service: str
) -> Dict[str, Any]:
    """
    Analyze function and generate documentation using specified service.
    
    Args:
        function_details: Dictionary containing function information
        service: Service to use ("azure", "openai", or "claude")
        
    Returns:
        Dictionary containing analysis results
    """
    # Define function_name early to avoid unbound variable issue
    function_name = function_details.get("name", "unknown")
    
    try:
        api_client = APIClient()
        analyzer = DocumentationAnalyzer(api_client)
        
        logger.info(f"Analyzing function: {function_name} using {service}")

        messages = [
            {
                "role": "system",
                "content": "You are an expert code documentation generator."
            },
            {
                "role": "user",
                "content": f"""Analyze and document this function:
                ```python
                {function_details.get('code', '')}
                ```
                """
            }
        ]

        response = await analyzer.make_api_request(messages, service)
        
        # Handle different response formats
        if service == "claude":
            content = response.content[0].text if isinstance(response.content, list) else response.content
            parsed_response = ClaudeResponseParser.parse_function_analysis(content)
        else:
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and tool_calls[0].function:
                function_args = json.loads(tool_calls[0].function.arguments)
                parsed_response = function_args
            else:
                return ClaudeResponseParser.get_default_response()

        # Ensure changelog is included in the parsed response
        if "changelog" not in parsed_response:
            parsed_response["changelog"] = []

        # Ensure classes is included in the parsed response
        if "classes" not in parsed_response:
            parsed_response["classes"] = []

        # Validate response against schema
        try:
            validate_schema(parsed_response)
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return ClaudeResponseParser.get_default_response()

        return {
            "name": function_name,
            "complexity_score": function_details.get("complexity_score", "Unknown"),
            "summary": parsed_response.get("summary", ""),
            "docstring": parsed_response.get("docstring", ""),
            "params": parsed_response.get("params", []),
            "returns": parsed_response.get("returns", {"type": "None", "description": ""}),
            "examples": parsed_response.get("examples", []),
            "classes": parsed_response.get("classes", []),
            "changelog": parsed_response.get("changelog")
        }

    except Exception as e:
        logger.error(f"Error analyzing function {function_name}: {e}")
        return ClaudeResponseParser.get_default_response()

class AsyncAPIClient:
    """
    Asynchronous API client for batch processing.
    Useful for processing multiple functions concurrently.
    """
    
    def __init__(self, service: str):
        self.service = service
        self.api_client = APIClient()
        self.analyzer = DocumentationAnalyzer(self.api_client)
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

    async def process_batch(
        self,
        functions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of functions concurrently.
        
        Args:
            functions: List of function details to process
            
        Returns:
            List of documentation results
        """
        async def process_with_semaphore(func: Dict[str, Any]) -> Dict[str, Any]:
            async with self.semaphore:
                return await analyze_function_with_openai(func, self.service)

        tasks = [process_with_semaphore(func) for func in functions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                processed_results.append(ClaudeResponseParser.get_default_response())
            else:
                processed_results.append(result)
                
        return processed_results

# Initialize default API client
default_api_client = APIClient()
