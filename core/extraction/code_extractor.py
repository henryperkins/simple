"""Main code extraction module."""

import ast
import re
import time
from typing import Optional, Dict, Any, Set, List
from core.logger import LoggerSetup
from core.metrics import Metrics
from .types import (
    ExtractionContext, ExtractionResult, ExtractedClass, 
    ExtractedFunction
)
from .utils import ASTUtils
from .function_extractor import FunctionExtractor
from .class_extractor import ClassExtractor
from .dependency_analyzer import DependencyAnalyzer

logger = LoggerSetup.get_logger(__name__)

class CodeExtractor:
    """Extracts code elements and metadata from Python source code."""

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """Initialize the code extractor."""
        self.logger = logger
        self.context = context or ExtractionContext()
        self._module_ast: Optional[ast.Module] = None
        self._current_class: Optional[ast.ClassDef] = None
        self.errors: List[str] = []
        self.metrics_calculator = Metrics()
        self.ast_utils = ASTUtils()
        self.function_extractor = FunctionExtractor(self.context, self.metrics_calculator)
        self.class_extractor = ClassExtractor(self.context, self.metrics_calculator)
        self.dependency_analyzer = DependencyAnalyzer(self.context)

    def extract_code(self, source_code: str, context: Optional[ExtractionContext] = None) -> Optional[ExtractionResult]:
        """Extract all code elements and metadata."""
        if context:
            self.context = context

        self.logger.info("Starting code extraction")
        start_time = time.time()

        try:
            processed_code = self._preprocess_code(source_code)
            tree = ast.parse(processed_code)
            self._module_ast = tree
            self.ast_utils.add_parents(tree)

            result = ExtractionResult(
                module_docstring=ast.get_docstring(tree)
            )

            # Extract dependencies
            try:
                result.dependencies = self.dependency_analyzer.analyze_dependencies(
                    tree,
                    self.context.module_name
                )
                self.logger.debug(f"Module dependencies: {result.dependencies}")
            except Exception as e:
                self._handle_extraction_error("Dependency analysis", e, result)

            # Extract code elements
            self._extract_elements(tree, result)

            # Calculate metrics if enabled
            if self.context.metrics_enabled:
                try:
                    self._calculate_metrics(result, tree)
                except Exception as e:
                    self._handle_extraction_error("Metrics calculation", e, result)

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
        """Preprocess source code to handle special cases."""
        try:
            pattern = r'\$\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?\$'
            processed_code = re.sub(pattern, r'"\g<0>"', source_code)
            self.logger.debug("Preprocessed source code to handle timestamps.")
            return processed_code
        except Exception as e:
            self.logger.error(f"Error preprocessing code: {e}", exc_info=True)
            return source_code

    def _extract_elements(self, tree: ast.AST, result: ExtractionResult) -> None:
        """Extract different code elements."""
        try:
            result.classes = self.class_extractor.extract_classes(tree)
            self.logger.debug(f"Extracted {len(result.classes)} classes.")
        except Exception as e:
            self._handle_extraction_error("Class extraction", e, result)

        try:
            result.functions = self.function_extractor.extract_functions(tree)
            self.logger.debug(f"Extracted {len(result.functions)} functions.")
        except Exception as e:
            self._handle_extraction_error("Function extraction", e, result)

        try:
            result.variables = self.ast_utils.extract_variables(tree)
            self.logger.debug(f"Extracted {len(result.variables)} variables.")
        except Exception as e:
            self._handle_extraction_error("Variable extraction", e, result)

        try:
            result.constants = self.ast_utils.extract_constants(tree)
            self.logger.debug(f"Extracted {len(result.constants)} constants.")
        except Exception as e:
            self._handle_extraction_error("Constant extraction", e, result)

        try:
            result.imports = self.dependency_analyzer.extract_imports(tree)
            self.logger.debug(f"Extracted imports: {result.imports}")
        except Exception as e:
            self._handle_extraction_error("Import extraction", e, result)
            result.imports = {'stdlib': set(), 'local': set(), 'third_party': set()}

    def _calculate_metrics(self, result: ExtractionResult, tree: ast.AST) -> None:
        """Calculate metrics for the extraction result."""
        if not self.context.metrics_enabled:
            return

        try:
            for cls in result.classes:
                self._calculate_class_metrics(cls)
            
            result.metrics.update(self._calculate_module_metrics(tree))

            for func in result.functions:
                self._calculate_function_metrics(func)

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}", exc_info=True)
            self.errors.append(str(e))

    def _calculate_class_metrics(self, cls: ExtractedClass) -> None:
        """
        Calculate metrics for a class.
        
        Args:
            cls (ExtractedClass): The extracted class to calculate metrics for.
        """
        try:
            if not cls.ast_node:
                return

            metrics = {
                'complexity': self.metrics_calculator.calculate_complexity(cls.ast_node),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(cls.ast_node),
                'method_count': len(cls.methods),
                'attribute_count': len(cls.attributes) + len(cls.instance_attributes)
            }
            
            cls.metrics.update(metrics)
            
        except Exception as e:
            self.logger.error(f"Error calculating class metrics: {e}")
        
    def _calculate_module_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Calculate metrics for the entire module.
        
        Args:
            tree (ast.AST): The module's AST.
            
        Returns:
            Dict[str, Any]: Module-level metrics.
        """
        try:
            return {
                'complexity': self.metrics_calculator.calculate_complexity(tree),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(tree),
                'lines': len(self.ast_utils.get_source_segment(tree).splitlines()) if self.ast_utils.get_source_segment(tree) else 0,
                'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                'functions': len([n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            }
        except Exception as e:
            self.logger.error(f"Error calculating module metrics: {e}")
            return {}
            
    def _calculate_function_metrics(self, func: ExtractedFunction) -> None:  
        """Calculate metrics for a given function."""  
        try:  
            # Assuming `func` is an instance of `ExtractedFunction`  
            metrics = {  
                'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(func.ast_node),  
                'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(func.ast_node),  
                'maintainability_index': self.metrics_calculator.calculate_maintainability_index(func.ast_node),  
                'parameter_count': len(func.args),  
                'return_complexity': self._calculate_return_complexity(func.ast_node),  
                'is_async': func.is_async  
            }  
            func.metrics.update(metrics)  
        except Exception as e:  
            self.logger.error(f"Error calculating function metrics: {e}", exc_info=True)  
            self.errors.append(str(e))  

    def _calculate_return_complexity(self, node: ast.AST) -> int:  
        """Calculate the complexity of return statements."""  
        try:  
            return sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))  
        except Exception as e:  
            self.logger.error(f"Error calculating return complexity: {e}", exc_info=True)  
            return 0
            
    def _handle_extraction_error(self, operation: str, error: Exception, result: ExtractionResult) -> None:
        """Handle extraction errors consistently."""
        error_msg = f"{operation} failed: {str(error)}"
        self.logger.warning(error_msg, exc_info=True)
        result.errors.append(error_msg)