"""
Extraction Manager Module

Manages the extraction of metadata from Python source code, focusing on class and function
definitions with integrated docstring processing and metrics calculation.
"""

import ast
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from core.logger import LoggerSetup
from core.utils import handle_exceptions
from core.metrics import Metrics
from core.docstring_processor import DocstringProcessor, DocstringData
from extract.ast_analysis import ASTAnalyzer
from extract.functions import FunctionExtractor
from extract.classes import ClassExtractor

logger = LoggerSetup.get_logger(__name__)

@dataclass
class ExtractionContext:
    """Context information for extraction process."""
    file_path: Optional[str] = None
    module_name: Optional[str] = None
    import_context: Optional[Dict[str, Set[str]]] = None
    metrics_enabled: bool = True
    include_source: bool = True

@dataclass
class ExtractionResult:
    """Structured result of extraction process."""
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    module_context: Dict[str, Any]
    metrics: Dict[str, Any]
    errors: List[str]

class ExtractionManager:
    """
    Manages extraction of metadata from Python source code, particularly classes and functions.
    Provides special handling for exception classes and integrated docstring processing.
    """

    def __init__(self, context: Optional[ExtractionContext] = None) -> None:
        """
        Initialize the ExtractionManager with analyzers and processors.

        Args:
            context: Optional extraction context configuration
        """
        logger.debug("Initializing ExtractionManager")
        self.analyzer = ASTAnalyzer()
        self.metrics_calculator = Metrics()
        self.docstring_processor = DocstringProcessor()
        self.context = context or ExtractionContext()
        self.errors: List[str] = []

    @handle_exceptions(logger.error)
    def extract_metadata(self, source_code: str) -> ExtractionResult:
        """
        Extract metadata from source code, including information about classes and functions.

        Args:
            source_code: The source code to analyze

        Returns:
            ExtractionResult: Structured extraction results including metrics
        """
        logger.debug("Starting metadata extraction")
        
        # Parse source code and add parent information
        tree = self.analyzer.parse_source_code(source_code)
        self.analyzer.add_parent_info(tree)

        # Extract classes and functions
        classes = self._extract_classes(tree, source_code)
        functions = self._extract_functions(tree, source_code)
        
        # Extract module context
        module_context = self._extract_module_context(tree)
        
        # Calculate module-level metrics
        metrics = self._calculate_module_metrics(tree, classes, functions)

        logger.info(
            f"Extraction complete. Found {len(classes)} classes and "
            f"{len(functions)} functions"
        )

        return ExtractionResult(
            classes=classes,
            functions=functions,
            module_context=module_context,
            metrics=metrics,
            errors=self.errors
        )

    def _extract_classes(
        self,
        tree: ast.AST,
        source_code: str
    ) -> List[Dict[str, Any]]:
        """Extract and process class definitions."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    # Process class with docstring
                    docstring_data = self.docstring_processor.process_node(node, source_code)
                    
                    # Extract class details
                    class_extractor = ClassExtractor(ast.unparse(node))
                    class_metadata = class_extractor.extract_details(node)
                    
                    # Calculate class metrics
                    metrics = self._calculate_class_metrics(node)
                    
                    # Combine all information
                    class_info = {
                        **class_metadata,
                        'docstring_data': docstring_data,
                        'metrics': metrics,
                        'source': ast.unparse(node) if self.context.include_source else None,
                        'is_exception': self._is_exception_class(node),
                        'dependencies': self._extract_dependencies(node)
                    }
                    
                    classes.append(class_info)
                    logger.debug(f"Extracted class: {node.name}")
                    
                except Exception as e:
                    error_msg = f"Error extracting class {node.name}: {str(e)}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
                    
        return classes

    def _extract_functions(
        self,
        tree: ast.AST,
        source_code: str
    ) -> List[Dict[str, Any]]:
        """Extract and process function definitions."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                try:
                    # Skip if function is a method (handled in class extraction)
                    if self._is_method(node):
                        continue
                        
                    # Process function with docstring
                    docstring_data = self.docstring_processor.process_node(node, source_code)
                    
                    # Extract function details
                    function_extractor = FunctionExtractor(ast.unparse(node))
                    function_metadata = function_extractor.extract_details(node)
                    
                    # Calculate function metrics
                    metrics = self._calculate_function_metrics(node)
                    
                    # Combine all information
                    function_info = {
                        **function_metadata,
                        'docstring_data': docstring_data,
                        'metrics': metrics,
                        'source': ast.unparse(node) if self.context.include_source else None,
                        'dependencies': self._extract_dependencies(node),
                        'complexity_warnings': self._get_complexity_warnings(metrics)
                    }
                    
                    functions.append(function_info)
                    logger.debug(f"Extracted function: {node.name}")
                    
                except Exception as e:
                    error_msg = f"Error extracting function {node.name}: {str(e)}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
                    
        return functions

    def _extract_module_context(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract module-level context and information."""
        logger.debug("Extracting module-level context")
        
        try:
            module_docstring = self.analyzer.extract_docstring(tree) or ''
            imports = self.analyzer.extract_imports(tree)
            global_vars = self.analyzer.extract_globals(tree)
            
            # Process module docstring
            if module_docstring:
                module_docstring_data = self.docstring_processor.parse(module_docstring)
            else:
                module_docstring_data = None

            context = {
                'module_docstring': module_docstring,
                'processed_docstring': module_docstring_data,
                'imports': imports,
                'global_variables': global_vars,
                'file_path': self.context.file_path,
                'module_name': self.context.module_name,
                'import_graph': self._analyze_import_graph(tree)
            }
            
            logger.debug(f"Extracted module context: {context}")
            return context
            
        except Exception as e:
            error_msg = f"Error extracting module context: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return {}

    def _calculate_module_metrics(
        self,
        tree: ast.AST,
        classes: List[Dict[str, Any]],
        functions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive module-level metrics."""
        if not self.context.metrics_enabled:
            return {}

        try:
            metrics = {
                'total_lines': len(ast.unparse(tree).splitlines()),
                'class_count': len(classes),
                'function_count': len(functions),
                'complexity': {
                    'average_cyclomatic': self._calculate_average_complexity(
                        classes, functions, 'cyclomatic_complexity'
                    ),
                    'average_cognitive': self._calculate_average_complexity(
                        classes, functions, 'cognitive_complexity'
                    ),
                    'maintainability_index': self.metrics_calculator.calculate_maintainability_index(tree)
                },
                'documentation': {
                    'documented_classes': sum(1 for c in classes if c.get('docstring_data')),
                    'documented_functions': sum(1 for f in functions if f.get('docstring_data')),
                    'documentation_coverage': self._calculate_documentation_coverage(
                        classes, functions
                    )
                }
            }
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error calculating module metrics: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return {}

    def _calculate_class_metrics(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Calculate metrics for a class."""
        if not self.context.metrics_enabled:
            return {}

        try:
            return {
                'method_count': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                'complexity': self.metrics_calculator.calculate_complexity(node),
                'maintainability': self.metrics_calculator.calculate_maintainability_index(node),
                'inheritance_depth': self._calculate_inheritance_depth(node)
            }
        except Exception as e:
            logger.error(f"Error calculating class metrics for {node.name}: {e}")
            return {}

    def _calculate_function_metrics(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate metrics for a function."""
        if not self.context.metrics_enabled:
            return {}

        try:
            return {
                'cyclomatic_complexity': self.metrics_calculator.calculate_cyclomatic_complexity(node),
                'cognitive_complexity': self.metrics_calculator.calculate_cognitive_complexity(node),
                'maintainability_index': self.metrics_calculator.calculate_maintainability_index(node),
                'parameter_count': len(node.args.args),
                'return_complexity': self._calculate_return_complexity(node)
            }
        except Exception as e:
            logger.error(f"Error calculating function metrics for {node.name}: {e}")
            return {}

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Determine if a function node is a method."""
        return any(
            isinstance(parent, ast.ClassDef)
            for parent in ast.walk(node)
            if hasattr(parent, 'body') and node in parent.body
        )

    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is an exception class."""
        return any(
            base.id in {'Exception', 'BaseException'}
            for base in node.bases
            if isinstance(base, ast.Name)
        )

    def _extract_dependencies(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies for a node."""
        try:
            return {
                'imports': self.analyzer.extract_imports(node),
                'calls': self._extract_function_calls(node),
                'attributes': self._extract_attribute_access(node)
            }
        except Exception as e:
            logger.error(f"Error extracting dependencies: {e}")
            return {}

    def _analyze_import_graph(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Analyze import relationships."""
        try:
            return {
                'direct_imports': set(self.analyzer.extract_imports(tree)),
                'indirect_imports': self._find_indirect_imports(tree)
            }
        except Exception as e:
            logger.error(f"Error analyzing import graph: {e}")
            return {}

    def _calculate_average_complexity(
        self,
        classes: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        metric_name: str
    ) -> float:
        """Calculate average complexity metric."""
        all_metrics = []
        
        # Get function metrics
        for func in functions:
            if metric_value := func.get('metrics', {}).get(metric_name):
                all_metrics.append(metric_value)
        
        # Get class method metrics
        for cls in classes:
            for method in cls.get('methods', []):
                if metric_value := method.get('metrics', {}).get(metric_name):
                    all_metrics.append(metric_value)
        
        return sum(all_metrics) / len(all_metrics) if all_metrics else 0.0

    def _calculate_documentation_coverage(
        self,
        classes: List[Dict[str, Any]],
        functions: List[Dict[str, Any]]
    ) -> float:
        """Calculate documentation coverage percentage."""
        total_items = len(classes) + len(functions)
        if not total_items:
            return 0.0
            
        documented_items = (
            sum(1 for c in classes if c.get('docstring_data')) +
            sum(1 for f in functions if f.get('docstring_data'))
        )
        
        return (documented_items / total_items) * 100

    def _get_complexity_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate warnings for high complexity."""
        warnings = []
        
        if metrics.get('cyclomatic_complexity', 0) > 10:
            warnings.append("High cyclomatic complexity")
        if metrics.get('cognitive_complexity', 0) > 15:
            warnings.append("High cognitive complexity")
        if metrics.get('maintainability_index', 100) < 20:
            warnings.append("Low maintainability index")
            
        return warnings

    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate inheritance depth of a class."""
        depth = 0
        bases = node.bases
        
        while bases:
            depth += 1
            new_bases = []
            for base in bases:
                if isinstance(base, ast.Name):
                    # This is a simplification; in a real implementation,
                    # you'd want to resolve the actual base classes
                    new_bases.extend(getattr(base, 'bases', []))
            bases = new_bases
            
        return depth

    def _calculate_return_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity of function's return statements."""
        return_count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return_count += 1
        return return_count

    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """Extract function calls from a node."""
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(f"{child.func.value.id}.{child.func.attr}")
        return calls

    def _extract_attribute_access(self, node: ast.AST) -> Set[str]:
        """Extract attribute access patterns."""
        attributes = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    attributes.add(f"{child.value.id}.{child.attr}")
        return attributes

    def _find_indirect_imports(self, tree: ast.AST) -> Set[str]:
        """Find indirect imports through imported modules."""
        # This is a placeholder for more complex import analysis
        return set()