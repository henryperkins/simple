import ast
import re
import time
from typing import Optional, List, Dict, Any, Union
from core.logger import LoggerSetup
from core.metrics import Metrics, MetricsCollector
from core.types import ExtractionContext, ExtractionResult
from core.function_extractor import FunctionExtractor
from core.class_extractor import ClassExtractor
from core.dependency_analyzer import DependencyAnalyzer
from utils import handle_extraction_error, get_source_segment
from docstringutils import DocstringUtils

logger = LoggerSetup.get_logger(__name__)

class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the CodeExtractor.

        Args:
            context (Optional[ExtractionContext]): The extraction context containing settings and configurations.
        """
        self.logger = logger
        self.context = context or ExtractionContext()
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        self.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        self.dependency_analyzer = DependencyAnalyzer(self.context)
        self.metrics_collector = MetricsCollector()

    async def extract_code_async(self, source_code: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Asynchronously extract code elements and metadata from source code.

        Args:
            source_code (str): The source code to be analyzed.
            context (Optional[ExtractionContext]): Optional context to override the existing one.

        Returns:
            Optional[ExtractionResult]: An object containing the extracted code elements and metrics.
        """
        start_time = time.time()
        success = False
        error_message = ""
        try:
            result = self.extract_code(source_code, context)
            success = True
            return result
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            duration = time.time() - start_time
            await self.metrics_collector.track_operation(
                operation_type="code_extraction",
                success=success,
                duration=duration,
                error=error_message if not success else None,
                metadata={
                    "module_name": self.context.module_name,
                }
            )

    def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata from source code.

        Args:
            source_code (str): The source code to be analyzed.
            context (Optional[ExtractionContext]): Optional context to override the existing one.

        Returns:
            Optional[ExtractionResult]: An object containing the extracted code elements and metrics.
        """
        if context:
            self.context = context
        self.context.source_code = source_code

        self.logger.info("Starting code extraction")
        start_time = time.time()

        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
            self._add_parent_references(tree)

            maintainability_index = self.metrics_calculator.calculate_maintainability_index(tree)

            result = ExtractionResult(
                module_docstring=DocstringUtils.extract_docstring_info(tree),
                maintainability_index=maintainability_index
            )

            try:
                result.dependencies = self.dependency_analyzer.analyze_dependencies(tree, self.context.module_name)
                self.logger.debug(f"Module dependencies: {result.dependencies}")
            except Exception as e:
                handle_extraction_error(self.logger, self.errors, "Dependency analysis", e)
                result.errors.extend(self.errors)

            self._extract_elements(tree, result)

            if self.context.metrics_enabled:
                try:
                    self._calculate_metrics(result)
                except Exception as e:
                    handle_extraction_error(self.logger, self.errors, "Metrics calculation", e)
                    result.errors.extend(self.errors)

            self.logger.info(f"Code extraction completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Extraction result: {len(result.classes)} classes, {len(result.functions)} functions")
            return result

        except SyntaxError as e:
            self.logger.error(f"Syntax error in source code: {str(e)}")
            return ExtractionResult(errors=[f"Syntax error: {str(e)}"])
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}", exc_info=True)
            return ExtractionResult(errors=[f"Failed to extract code: {str(e)}"])

    def _preprocess_code(self, source_code: str) -> str:
        """Preprocess source code to handle special cases.

        Args:
            source_code (str): The source code to preprocess.

        Returns:
            str: The preprocessed source code.
        """
        try:
            pattern = r'\$\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?'
            processed_code = re.sub(pattern, r'"\g<0>"', source_code)
            self.logger.debug("Preprocessed source code to handle timestamps.")
            return processed_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}", exc_info=True)
            return source_code

    def _add_parent_references(self, node: ast.AST) -> None:
        """Add parent references to AST nodes.

        Args:
            node (ast.AST): The root node of the AST.
        """
        for child in ast.walk(node):
            for child_node in ast.iter_child_nodes(child):
                setattr(child_node, 'parent', child)

    def _extract_elements(self, tree: ast.AST, result: ExtractionResult) -> None:
        """Extract different code elements from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.
            result (ExtractionResult): The result object to store extracted elements.
        """
        try:
            result.classes = self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Class extraction", e)
            result.errors.extend(self.errors)

        try:
            result.functions = self.function_extractor.extract_functions(tree)
            self.logger.debug(f"Extracted {len(result.functions)} functions.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Function extraction", e)
            result.errors.extend(self.errors)

        try:
            result.variables = self._extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Variable extraction", e)
            result.errors.extend(self.errors)

        try:
            result.constants = self._extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")
        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Constant extraction", e)
            result.errors.extend(self.errors)

    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variables from the AST.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing variable information.
        """
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        var_info = self._create_variable_info(target, node)
                        if var_info:
                            variables.append(var_info)
        return variables

    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants.

        Args:
            tree (ast.AST): The root of the AST representing the parsed Python source code.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing constant information.
        """
        constants = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constant_info = self._create_constant_info(target, node)
                        if constant_info:
                            constants.append(constant_info)
        return constants

    def _create_variable_info(self, target: ast.Name, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[Dict[str, Any]]:
        """Create variable information dictionary.

        Args:
            target (ast.Name): The target node representing the variable.
            node (Union[ast.Assign, ast.AnnAssign]): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing variable information or None if an error occurs.
        """
        try:
            var_name = target.id
            annotation = None
            value = None

            if isinstance(node, ast.AnnAssign) and node.annotation:
                annotation = DocstringUtils.get_node_name(node.annotation)
            if hasattr(node, 'value') and node.value:
                try:
                    value = DocstringUtils.get_node_name(node.value)
                except Exception as e:
                    logger.error(f"Failed to get value for {var_name}: {e}")
                    value = "UnknownValue"

            return {
                'name': var_name,
                'type': annotation or "UnknownType",
                'value': value
            }
        except Exception as e:
            logger.error(f"Error creating variable info: {e}")
            return None

    def _create_constant_info(self, target: ast.Name, node: ast.Assign) -> Optional[Dict[str, Any]]:
        """Create constant information dictionary.

        Args:
            target (ast.Name): The target node representing the constant.
            node (ast.Assign): The assignment node.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing constant information or None if an error occurs.
        """
        try:
            value = DocstringUtils.get_node_name(node.value)
            try:
                value_type = type(ast.literal_eval(node.value)).__name__
            except Exception:
                value_type = "UnknownType"
            return {
                'name': target.id,
                'value': value,
                'type': value_type
            }
        except Exception as e:
            logger.error(f"Error creating constant info: {e}")
            return None

    def _calculate_metrics(self, result: ExtractionResult) -> None:
        """Calculate metrics for the extraction result.

        Args:
            result (ExtractionResult): The result object containing extracted elements.
        """
        try:
            for cls in result.classes:
                cls.metrics = self.metrics_calculator.calculate_class_metrics(cls.ast_node)

            for func in result.functions:
                func.metrics = self.metrics_calculator.calculate_function_metrics(func.ast_node)

        except Exception as e:
            handle_extraction_error(self.logger, self.errors, "Metrics calculation", e)
            result.errors.extend(self.errors)