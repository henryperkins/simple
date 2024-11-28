# core/docstring_processor.py
"""
Core docstring processing module with integrated metrics and extraction capabilities.
"""

import ast
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from core.logger import LoggerSetup
from core.metrics import Metrics
from extract.ast_analysis import ASTAnalyzer

logger = LoggerSetup.get_logger(__name__)

@dataclass
class DocstringMetrics:
    """Metrics for docstring quality and complexity."""
    length: int
    sections_count: int
    args_count: int
    cognitive_complexity: float
    completeness_score: float

@dataclass
class DocstringData:
    """Enhanced docstring data with metrics."""
    summary: str
    description: str
    args: List[Dict[str, str]]
    returns: Dict[str, str]
    raises: Optional[List[Dict[str, str]]] = None
    metrics: Optional[DocstringMetrics] = None
    extraction_context: Optional[Dict[str, Any]] = None

class DocstringProcessor:
    """Enhanced docstring processor with metrics and extraction integration."""

    def __init__(self):
        self.metrics_calculator = Metrics()
        self.ast_analyzer = ASTAnalyzer()
        self.min_length = {
            'summary': 10,
            'description': 10
        }

    def process_node(
        self,
        node: ast.AST,
        source_code: str
    ) -> DocstringData:
        """
        Process an AST node to extract and analyze docstring information.

        Args:
            node: AST node to process
            source_code: Original source code for context

        Returns:
            DocstringData: Processed docstring data with metrics
        """
        # Extract existing docstring and context
        existing_docstring = ast.get_docstring(node) or ''
        extraction_context = self._extract_context(node, source_code)
        
        # Parse docstring
        docstring_data = self.parse(existing_docstring)
        
        # Calculate metrics
        metrics = self._calculate_metrics(docstring_data, node)
        
        # Enhance docstring data with context and metrics
        docstring_data.metrics = metrics
        docstring_data.extraction_context = extraction_context
        
        return docstring_data

    def _extract_context(
        self,
        node: ast.AST,
        source_code: str
    ) -> Dict[str, Any]:
        """Extract context information for the node."""
        context = {}
        
        if isinstance(node, ast.FunctionDef):
            context.update({
                'type': 'function',
                'name': node.name,
                'args': self._extract_function_args(node),
                'returns': self.ast_analyzer.get_annotation(node.returns),
                'complexity': self.metrics_calculator.calculate_complexity(node),
                'source': ast.unparse(node)
            })
            
        elif isinstance(node, ast.ClassDef):
            context.update({
                'type': 'class',
                'name': node.name,
                'bases': [self.ast_analyzer.get_annotation(base) for base in node.bases],
                'methods': self._extract_class_methods(node),
                'source': ast.unparse(node)
            })
            
        return context

    def _extract_function_args(
        self,
        node: ast.FunctionDef
    ) -> List[Dict[str, str]]:
        """Extract function arguments with type annotations."""
        args = []
        for arg in node.args.args:
            args.append({
                'name': arg.arg,
                'type': self.ast_analyzer.get_annotation(arg.annotation),
                'default': self._get_default_value(arg, node)
            })
        return args

    def _extract_class_methods(
        self,
        node: ast.ClassDef
    ) -> List[Dict[str, Any]]:
        """Extract class methods with their signatures."""
        methods = []
        for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
            methods.append({
                'name': method.name,
                'args': self._extract_function_args(method),
                'returns': self.ast_analyzer.get_annotation(method.returns),
                'is_property': any(isinstance(d, ast.Name) and d.id == 'property'
                                 for d in method.decorator_list)
            })
        return methods

    def _calculate_metrics(
        self,
        docstring_data: DocstringData,
        node: ast.AST
    ) -> DocstringMetrics:
        """Calculate comprehensive metrics for the docstring."""
        # Basic metrics
        length = len(docstring_data.summary) + len(docstring_data.description)
        sections_count = sum(1 for x in [
            docstring_data.summary,
            docstring_data.description,
            docstring_data.args,
            docstring_data.returns,
            docstring_data.raises
        ] if x)
        args_count = len(docstring_data.args)

        # Calculate cognitive complexity
        if isinstance(node, ast.FunctionDef):
            cognitive_complexity = self.metrics_calculator.calculate_cognitive_complexity(node)
        else:
            cognitive_complexity = 0.0

        # Calculate completeness score
        completeness_score = self._calculate_completeness(docstring_data, node)

        return DocstringMetrics(
            length=length,
            sections_count=sections_count,
            args_count=args_count,
            cognitive_complexity=cognitive_complexity,
            completeness_score=completeness_score
        )

    def _calculate_completeness(
        self,
        docstring_data: DocstringData,
        node: ast.AST
    ) -> float:
        """Calculate docstring completeness score."""
        score = 0.0
        total_checks = 0

        # Check summary
        if len(docstring_data.summary) >= self.min_length['summary']:
            score += 1
        total_checks += 1

        # Check description
        if len(docstring_data.description) >= self.min_length['description']:
            score += 1
        total_checks += 1

        # Check arguments documentation
        if isinstance(node, ast.FunctionDef):
            actual_args = {arg.arg for arg in node.args.args}
            documented_args = {arg['name'] for arg in docstring_data.args}
            
            if actual_args == documented_args:
                score += 1
            total_checks += 1

            # Check return value documentation
            if node.returns and docstring_data.returns['type'] != 'None':
                score += 1
            total_checks += 1

        return (score / total_checks) * 100 if total_checks > 0 else 0.0

    def validate(
        self,
        data: DocstringData,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate docstring with enhanced context awareness.

        Args:
            data: Docstring data to validate
            context: Optional extraction context for additional validation

        Returns:
            Tuple of validation status and error messages
        """
        errors = []
        
        # Basic validation
        if len(data.summary.strip()) < self.min_length['summary']:
            errors.append("Summary too short")
            
        # Context-aware validation
        if context:
            if context['type'] == 'function':
                # Validate function arguments
                actual_args = {arg['name'] for arg in context['args']}
                documented_args = {arg['name'] for arg in data.args}
                
                missing_args = actual_args - documented_args
                if missing_args:
                    errors.append(f"Missing documentation for arguments: {missing_args}")
                
                extra_args = documented_args - actual_args
                if extra_args:
                    errors.append(f"Documentation for non-existent arguments: {extra_args}")
                
                # Validate return type
                if context['returns'] != 'None' and not data.returns['type']:
                    errors.append("Missing return value documentation")

        return len(errors) == 0, errors

    def _get_default_value(
        self,
        arg: ast.arg,
        func_node: ast.FunctionDef
    ) -> Optional[str]:
        """Get default value for a function argument."""
        try:
            defaults = func_node.args.defaults
            if defaults:
                args = func_node.args.args
                default_index = len(args) - len(defaults)
                arg_index = args.index(arg)
                
                if arg_index >= default_index:
                    default_node = defaults[arg_index - default_index]
                    return ast.unparse(default_node)
        except Exception as e:
            logger.error(f"Error getting default value: {e}")
        return None