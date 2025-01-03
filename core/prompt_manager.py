from typing import Optional, Any, Dict
from pathlib import Path
import json
from jinja2 import (
    Template,
    TemplateError,
    Environment,
    FileSystemLoader,
    TemplateNotFound,
)
import time

from core.types.base import (
    ExtractedClass,
    ExtractedFunction,
    ProcessingResult,
    MetricData,
    TokenUsage,
    DocumentationContext,
    ExtractedArgument,
)
from core.logger import CorrelationLoggerAdapter, LoggerSetup
from core.metrics_collector import MetricsCollector
from core.console import print_info
from core.types.docstring import DocstringData
from core.exceptions import PromptGenerationError, TemplateLoadingError
from utils import log_and_raise_error

class PromptManager:
    documentation_template: (
        Template  # Explicitly declare the type of documentation_template
    )
    env: Environment  # Explicitly declare the type of env

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the PromptManager with template loading and configuration."""
        from core.dependency_injection import Injector

        self.correlation_id = correlation_id
        self.logger = CorrelationLoggerAdapter(
            LoggerSetup.get_logger(__name__), extra={"correlation_id": correlation_id}
        )
        self.metrics_collector = MetricsCollector(correlation_id=correlation_id)
        self.token_manager = Injector.get("token_manager")

        # Define schema directory
        schema_dir = Path(__file__).parent / "schemas"

        # Load function schema
        try:
            self._function_schema: dict[str, Any] = self._load_and_validate_schema(
                schema_dir / "function_tools_schema.json"
            )
            if not self._function_schema or "function" not in self._function_schema:
                raise ValueError("Invalid function schema structure")
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Failed to load function schema",
                self.correlation_id,
            )
            self._function_schema = {
                "function": {
                    "name": "generate_docstring",
                    "description": "Generate documentation from source code",
                    "parameters": {},
                }
            }

        # Load templates using Jinja2 with enhanced error handling
        template_dir = Path(__file__).parent
        try:
            self.env: Environment = Environment(
                loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )

            # Add template helper functions
            self.env.globals.update(
                {
                    "_format_class_info": self._format_class_info,
                    "_format_function_info": self._format_function_info,
                }
            )

            # Load our specific template files
            self.documentation_template = self._load_template(
                "documentation_prompt.txt"
            )
            self.code_analysis_template = self._load_template(
                "code_analysis_prompt.txt"
            )

            self.logger.info("Templates loaded successfully")
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                TemplateLoadingError,
                "Template loading failed",
                self.correlation_id,
            )

    def _load_and_validate_schema(self, schema_path: Path) -> dict[str, Any]:
        """Load and validate a JSON schema with enhanced error handling."""
        try:
            with schema_path.open("r", encoding="utf-8") as f:
                schema = json.load(f)

            # Validate schema structure
            required_keys = ["type", "function"]
            if not all(key in schema for key in required_keys):
                raise ValueError(f"Schema missing required keys: {required_keys}")

            return schema
        except json.JSONDecodeError as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Invalid JSON in schema file",
                self.correlation_id,
                schema_path=str(schema_path),
            )
            raise
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Schema loading error",
                self.correlation_id,
                schema_path=str(schema_path),
            )
            raise

    def _load_template(self, template_name: str) -> Template:
        """
        Load and validate a template file.

        This method loads template files and performs basic validation to ensure
        they contain the expected sections and placeholders.

        Args:
            template_name: The name of the template file.

        Returns:
            The loaded template as a Jinja2 Template object.
        """
        try:
            template = self.env.get_template(template_name)

            # Validate template content
            rendered: str = template.render(
                {
                    "code": "TEST_CODE",
                    "module_name": "TEST_MODULE",
                    "file_path": "TEST_PATH",
                }
            )

            if not rendered or len(rendered) < 100:
                raise TemplateLoadingError(
                    f"Template {template_name} appears to be empty or invalid"
                )

            return template

        except TemplateNotFound as e:
            log_and_raise_error(
                self.logger,
                e,
                TemplateLoadingError,
                f"Template file not found: {template_name}",
                self.correlation_id,
            )
            raise
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                TemplateLoadingError,
                f"Error loading template {template_name}",
                self.correlation_id,
            )
            raise

    def _format_class_info(self, cls: ExtractedClass) -> str:
        """Format class information for template rendering."""
        output = [f"Class: {cls.name}"]

        docstring_info = cls.docstring_info
        if docstring_info:
            output.append(f"Description: {docstring_info.summary}")

        if cls.bases:
            output.append(f"Inherits from: {', '.join(cls.bases)}")

        if cls.methods:
            output.append("\nMethods:")
            for method in cls.methods:
                output.append(f"\n  {method.name}")
                method_docstring = method.get_docstring_info()
                if method_docstring:
                    output.append(f"  Description: {method_docstring.summary}")
                if method.args:
                    args_str = ", ".join(
                        f"{arg.name}: {arg.type or 'Any'}" for arg in method.args
                    )
                    output.append(f"  Arguments: {args_str}")
                if method.returns:
                    output.append(f"  Returns: {method.returns}")

        if cls.metrics:
            metrics = cls.metrics
            if isinstance(metrics, MetricData):
                output.append("\nMetrics:")
                output.append(f"  Complexity: {metrics.cyclomatic_complexity}")
                output.append(f"  Maintainability: {metrics.maintainability_index}")
                output.append(f"  Lines of Code: {metrics.lines_of_code}")
            elif isinstance(metrics, dict):
                output.append("\nMetrics:")
                output.append(f"  Complexity: {metrics.get('complexity', 'N/A')}")
                output.append(
                    f"  Maintainability: {metrics.get('maintainability', 'N/A')}"
                )
                output.append(f"  Lines of Code: {metrics.get('lines_of_code', 'N/A')}")

        return "\n".join(output)

    def _format_function_info(self, func: ExtractedFunction) -> str:
        """Format function information for template rendering."""
        output = [f"Function: {func.name}"]

        docstring_info = func.get_docstring_info()
        if docstring_info:
            output.append(f"Description: {docstring_info.summary}")

        if func.args:
            args_str = ", ".join(
                f"{arg.name}: {arg.type or 'Any'}" for arg in func.args
            )
            output.append(f"Arguments: {args_str}")

            # Add argument descriptions if available
            for arg in func.args:
                if arg.description:
                    output.append(f"  {arg.name}: {arg.description}")

        return "\n".join(output)

    async def create_documentation_prompt(
        self,
        context: DocumentationContext,
    ) -> ProcessingResult:
        """
        Create a documentation prompt using the documentation template.

        Args:
            context: Structured context containing all necessary documentation information.

        Returns:
            ProcessingResult containing the generated prompt and associated metrics.
        """
        print_info("Generating documentation prompt using template.")
        start_time = time.time()

        try:
            # Prepare template variables with explicit typing
            template_vars: Dict[str, Any] = {
                "module_name": context.metadata.get("module_name", ""),
                "file_path": str(context.module_path),
                "source_code": context.source_code,
                "classes": context.classes,
                "functions": context.functions,
            }

            # Render the template with explicit type hints
            prompt: str = self.documentation_template.render(**template_vars)

            # Track token usage
            token_usage = await self._calculate_token_usage(prompt)

            print(f"----- DEBUG: Generated Prompt Length (tokens): {token_usage.prompt_tokens}")
            print(f"----- DEBUG: Generated Prompt:\n{prompt}")

            # Track metrics
            metrics = await self._create_metrics(prompt, start_time)

            return ProcessingResult(
                content={"prompt": prompt},
                usage=token_usage.__dict__,
                metrics=metrics.__dict__,
                validation_status=True,
                validation_errors=[],
                schema_errors=[],
            )

        except TemplateError as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Template rendering failed",
                self.correlation_id,
            )
            return ProcessingResult(
                content={},
                usage={},
                metrics={},
                validation_status=False,
                validation_errors=[f"Template rendering failed: {str(e)}"],
                schema_errors=[],
            )
        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Error generating documentation prompt",
                self.correlation_id,
            )
            return ProcessingResult(
                content={},
                usage={},
                metrics={},
                validation_status=False,
                validation_errors=[str(e)],
                schema_errors=[],
            )

    def _format_argument(self, arg: ExtractedArgument) -> Dict[str, Any]:
        """Format ExtractedArgument instance."""
        return {
            "name": arg.name,
            "type": arg.type or "Any",
            "default_value": arg.default_value,
            "description": arg.description or "",
        }

    def _format_metrics(self, metrics: MetricData | Dict[str, Any]) -> Dict[str, Any]:
        """Format metrics data for template rendering."""
        if isinstance(metrics, MetricData):
            return {
                "complexity": metrics.cyclomatic_complexity,
                "maintainability": metrics.maintainability_index,
                "lines_of_code": metrics.lines_of_code,
                "scanned_functions": metrics.scanned_functions,
                "total_functions": metrics.total_functions,
                "function_scan_ratio": metrics.function_scan_ratio,
            }
        return metrics  # Return as-is if already a dict

    # In core/prompt_manager.py
    async def _calculate_token_usage(self, prompt: str) -> TokenUsage:
        """Calculate token usage for the prompt."""
        prompt_tokens = self.token_manager.estimate_tokens(prompt)
        
        # Get token costs from the model config
        costs = self.token_manager.get_token_costs()
        cost_per_token = costs["prompt_cost_per_1k"] / 1000  # Convert from per 1k tokens to per token
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,  # Will be filled later
            total_tokens=prompt_tokens,
            estimated_cost=cost_per_token * prompt_tokens
        )

    async def _create_metrics(self, prompt: str, start_time: float) -> MetricData:
        """Create MetricData for the prompt generation process."""
        return MetricData(
            module_name=self.correlation_id or "unknown",
            cyclomatic_complexity=1,  # Prompt complexity
            cognitive_complexity=1,
            maintainability_index=100.0,
            halstead_metrics={},
            lines_of_code=len(prompt.splitlines()),
            total_functions=0,
            scanned_functions=0,
            function_scan_ratio=0.0,
            total_classes=0,
            scanned_classes=0,
            class_scan_ratio=0.0,
        )

    def _format_docstring(self, docstring: Optional[DocstringData]) -> Dict[str, Any]:
        """Format DocstringData for template rendering."""
        if not docstring:
            return {
                "summary": "No description available.",
                "description": "No detailed description available.",
                "args": [],
                "returns": {"type": "Any", "description": ""},
                "raises": [],
                "complexity": 1,
            }

        return docstring.to_dict()

    def get_prompt_with_schema(self, prompt: str, schema: dict[str, Any]) -> str:
        """
        Adds function calling instructions to a prompt.

        Args:
            prompt: The base prompt.
            schema: The schema to use for function calling.

        Returns:
            The prompt with function calling instructions.
        """
        self.logger.debug("Adding function calling instructions to prompt")
        return f"{prompt}\n\nPlease respond with a JSON object that matches the schema defined in the function parameters."

    def get_function_schema(
        self, schema: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Get the function schema for structured output.

        Returns:
            Function schema dictionary.

        Raises:
            ValueError: If the schema is not properly formatted.
        """
        self.logger.debug("Retrieving function schema")

        if schema:
            return {
                "name": "generate_docstring",
                "description": "Generates structured documentation from source code.",
                "parameters": schema,
            }

        if not hasattr(self, "_function_schema") or not self._function_schema:
            raise ValueError("Function schema is not properly defined.")

        return self._function_schema["function"]

    async def create_code_analysis_prompt(
        self, source_code: str, context: Optional[str] = None
    ) -> ProcessingResult:
        """
        Create a code analysis prompt using the code analysis template.

        Args:
            source_code: The source code to analyze
            context: Optional context for the analysis

        Returns:
            ProcessingResult containing the generated prompt and associated metrics
        """
        start_time = time.time()

        try:
            # Render the template with our code and context
            prompt = self.code_analysis_template.render(
                code=source_code,
                context=context
                or "This code is part of a documentation generation system.",
            )

            # Track token usage
            token_usage = await self._calculate_token_usage(prompt)

            # Track metrics
            metrics = await self._create_metrics(prompt, start_time)

            return ProcessingResult(
                content={"prompt": prompt},
                usage=token_usage.__dict__,
                metrics=metrics.__dict__,
                validation_status=True,
                validation_errors=[],
                schema_errors=[],
            )

        except Exception as e:
            log_and_raise_error(
                self.logger,
                e,
                PromptGenerationError,
                "Error generating code analysis prompt",
                self.correlation_id,
            )
            return ProcessingResult(
                content={},
                usage={},
                metrics={},
                validation_status=False,
                validation_errors=[str(e)],
                schema_errors=[],
            )
