"""
AI Interaction Handler Module

Manages interactions with Azure OpenAI API using centralized response parsing.
"""

import asyncio
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import ast
import openai
from openai import AsyncAzureOpenAI
from core.logger import LoggerSetup
from core.cache import Cache
from core.metrics import MetricsCollector, Metrics
from core.config import AzureOpenAIConfig
from core.extraction.code_extractor import CodeExtractor
from core.docstring_processor import DocstringProcessor
from core.extraction.types import ExtractionContext
from core.types import ProcessingResult, AIHandler, DocumentationContext
from api.token_management import TokenManager
from api.api_client import APIClient
from core.response_parsing import ResponseParsingService
from exceptions import ValidationError, ExtractionError

logger = LoggerSetup.get_logger(__name__)

class AIInteractionHandler:
    def __init__(
        self, 
        config: Optional[AzureOpenAIConfig] = None,
        cache: Optional[Cache] = None,
        token_manager: Optional[TokenManager] = None,
        code_extractor: Optional[CodeExtractor] = None,
        response_parser: Optional[ResponseParsingService] = None,
        metrics: Optional[Metrics] = None
    ) -> None:
        self.logger = LoggerSetup.get_logger(__name__)
        try:
            # Initialize metrics first
            self.metrics = metrics or Metrics()
            
            # Create extraction context
            self.context = ExtractionContext(
                metrics=self.metrics,
                metrics_enabled=True,
                include_private=False,
                include_magic=False
            )
            
            # Initialize components with context
            self.config = config or AzureOpenAIConfig.from_env()
            self.cache = cache
            self.token_manager = token_manager or TokenManager(
                model=self.config.model_name,
                deployment_name=self.config.deployment_id,
                config=self.config
            )
            self.code_extractor = code_extractor or CodeExtractor(context=self.context)
            self.docstring_processor = DocstringProcessor(metrics=self.metrics)
            self.response_parser = response_parser or ResponseParsingService()
            
            # Initialize API client
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AIInteractionHandler: {e}")
            raise

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code before parsing."""
        try:
            processed_code = source_code.strip()
            self.logger.debug("Preprocessed source code")
            return processed_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}")
            return source_code

    async def process_code(
        self,
        source_code: str,
        cache_key: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        context: Optional[DocumentationContext] = None
    ) -> Optional[Tuple[str, str]]:
        """
        Process source code to generate documentation.

        Args:
            source_code: The source code to process
            cache_key: Optional cache key for storing results
            extracted_info: Optional pre-extracted code information
            context: Optional extraction context

        Returns:
            Optional[Tuple[str, str]]: Tuple of (updated_code, documentation) or None if processing fails

        Raises:
            ExtractionError: If code extraction fails
            ValidationError: If response validation fails
        """
        try:
            # Check cache first if enabled
            if cache_key and self.cache:
                try:
                    cached_result = await self.cache.get_cached_docstring(cache_key)
                    if cached_result:
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        code = cached_result.get("updated_code")
                        docs = cached_result.get("documentation")
                        return (code, docs) if isinstance(code, str) and isinstance(docs, str) else None
                except Exception as e:
                    self.logger.error(f"Cache retrieval error: {e}")

            # Process and validate source code
            processed_code = self._preprocess_code(source_code)
            try:
                tree = ast.parse(processed_code)
            except SyntaxError as e:
                self.logger.error(f"Syntax error in source code: {e}")
                raise ExtractionError(f"Failed to parse code: {e}")

            # Extract metadata if not provided
            if not extracted_info:
                ctx = self.context
                if isinstance(context, DocumentationContext):
                    ctx = ExtractionContext(
                        metrics=self.metrics,
                        metrics_enabled=True,
                        include_private=False,
                        include_magic=False
                    )
                extraction_result = self.code_extractor.extract_code(
                    processed_code,
                    ctx
                )
                if not extraction_result:
                    raise ExtractionError("Failed to extract code information")
                extracted_info = {
                    'module_docstring': extraction_result.module_docstring,
                    'metrics': extraction_result.metrics
                }

            # Generate prompt using existing function_calling_prompt
            try:
                prompt = self._create_function_calling_prompt(processed_code, extracted_info)
                completion = await self.client.chat.completions.create(
                    model=self.config.deployment_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
            except Exception as e:
                self.logger.error(f"Error during Azure OpenAI API call: {e}")
                raise

            # Use ResponseParsingService to parse and validate the response
            content = completion.choices[0].message.content
            if content is None:
                raise ValidationError("Empty response from AI service")
            
            parsed_response = await self.response_parser.parse_response(
                response=content,
                expected_format='docstring',
                validate_schema=True
            )

            # If parsing was successful, use DocstringProcessor to handle the docstrings
            if parsed_response.validation_success:
                try:
                    # Parse the docstring data using DocstringProcessor
                    docstring_data = self.docstring_processor.parse(parsed_response.content)
                    
                    # Ensure the AI-generated documentation matches the expected structure
                    ai_documentation = {
                        'summary': docstring_data.summary or "No summary provided",
                        'description': docstring_data.description or "No description provided",
                        'args': docstring_data.args or [],
                        'returns': docstring_data.returns or {'type': 'Any', 'description': ''},
                        'raises': docstring_data.raises or [],
                        'complexity': docstring_data.complexity or 1
                    }

                    # Create AST transformer to handle all node types
                    class DocstringTransformer(ast.NodeTransformer):
                        def __init__(self, docstring_processor, docstring_data):
                            self.docstring_processor = docstring_processor
                            self.docstring_data = docstring_data

                        def visit_Module(self, node):
                            # Handle module-level docstring
                            if self.docstring_data.summary:
                                module_docstring = self.docstring_processor.format(self.docstring_data)
                                node = self.docstring_processor.insert_docstring(node, module_docstring)
                            return self.generic_visit(node)

                        def visit_ClassDef(self, node):
                            # Handle class docstrings
                            if hasattr(self.docstring_data, 'classes') and \
                            node.name in self.docstring_data.classes:
                                class_data = self.docstring_data.classes[node.name]
                                class_docstring = self.docstring_processor.format(class_data)
                                node = self.docstring_processor.insert_docstring(node, class_docstring)
                            return self.generic_visit(node)

                        def visit_FunctionDef(self, node):
                            # Handle function docstrings
                            if hasattr(self.docstring_data, 'functions') and \
                            node.name in self.docstring_data.functions:
                                func_data = self.docstring_data.functions[node.name]
                                func_docstring = self.docstring_processor.format(func_data)
                                node = self.docstring_processor.insert_docstring(node, func_docstring)
                            return self.generic_visit(node)

                    # Apply the transformer
                    transformer = DocstringTransformer(self.docstring_processor, docstring_data)
                    modified_tree = transformer.visit(tree)
                    ast.fix_missing_locations(modified_tree)

                    # Convert back to source code
                    updated_code = ast.unparse(modified_tree)
                    
                    # Format the documentation using DocstringProcessor
                    documentation = self.docstring_processor.format(docstring_data)
                    
                    # Update the context with AI-generated documentation
                    if context:
                        context.ai_generated = ai_documentation

                    # Cache the result if caching is enabled
                    if cache_key and self.cache:
                        try:
                            await self.cache.store_docstring(
                                cache_key,
                                {
                                    "updated_code": updated_code,
                                    "documentation": documentation
                                }
                            )
                        except Exception as e:
                            self.logger.error(f"Cache storage error: {e}")

                    return updated_code, documentation

                except Exception as e:
                    self.logger.error(f"Error processing docstrings: {e}")
                    return None
            else:
                self.logger.warning(f"Response parsing had errors: {parsed_response.errors}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing code: {e}")
            return None
        
    def _insert_docstrings(self, tree: ast.AST, docstrings: Dict[str, Any]) -> ast.AST:
        """Insert docstrings into the AST."""
        class DocstringTransformer(ast.NodeTransformer):
            def visit_ClassDef(self, node):
                # Insert class docstring
                if node.name in docstrings:
                    docstring = ast.Constant(value=docstrings[node.name])
                    node.body.insert(0, ast.Expr(value=docstring))
                return node

            def visit_FunctionDef(self, node):
                # Insert function docstring
                if node.name in docstrings:
                    docstring = ast.Constant(value=docstrings[node.name])
                    node.body.insert(0, ast.Expr(value=docstring))
                return node

        transformer = DocstringTransformer()
        return transformer.visit(tree)
    
    def _create_function_calling_prompt(self, source_code: str, metadata: Dict[str, Any]) -> str:
        """Create the prompt for function calling with schema-compliant JSON output."""
        return (
            "Generate documentation for the provided code as a JSON object.\n\n"
            "REQUIRED OUTPUT FORMAT:\n"
            "```json\n"
            "{\n"
            '  "summary": "A brief one-line summary of the function/method",\n'
            '  "description": "Detailed description of the functionality",\n'
            '  "args": [\n'
            '    {\n'
            '      "name": "string - parameter name",\n'
            '      "type": "string - parameter data type",\n'
            '      "description": "string - brief description of the parameter"\n'
            '    }\n'
            '  ],\n'
            '  "returns": {\n'
            '    "type": "string - return data type",\n'
            '    "description": "string - brief description of return value"\n'
            '  },\n'
            '  "raises": [\n'
            '    {\n'
            '      "exception": "string - exception class name",\n'
            '      "description": "string - circumstances under which raised"\n'
            '    }\n'
            '  ],\n'
            '  "complexity": "integer - McCabe complexity score"\n'
            '}\n'
            '```\n\n'
            "VALIDATION REQUIREMENTS:\n"
            "1. All fields shown above are required\n"
            "2. All strings must be descriptive and clear\n"
            "3. Types must be accurate Python types\n"
            "4. Complexity must be a positive integer\n"
            "5. If complexity > 10, note this in the description with [WARNING]\n\n"
            "CODE TO DOCUMENT:\n"
            "```python\n"
            f"{source_code}\n"
            "```\n\n"
            "IMPORTANT:\n"
            "1. Always include a 'complexity' field with an integer value\n"
            "2. If complexity cannot be determined, use 1 as default\n"
            "3. Never set complexity to null or omit it\n\n"
            "Respond with only the JSON object. Do not include any other text."
        )
    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            if self.client:
                await self.client.close()
            if self.cache:
                await self.cache.close()
        except Exception as e:
            self.logger.error(f"Error closing AI handler: {e}")
            raise

    async def __aenter__(self) -> "AIInteractionHandler":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()