import ast
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionMetadata:
    name: str
    args: List[Dict[str, Optional[str]]]
    return_type: Optional[str]
    decorators: List[str]
    docstring: Optional[str]

@dataclass
class ClassMetadata:
    name: str
    base_classes: List[str]
    methods: List[FunctionMetadata]
    docstring: Optional[str]

@dataclass
class ModuleMetadata:
    name: str
    docstring: Optional[str]

@dataclass
class ExtractionResult:
    module_metadata: ModuleMetadata
    classes: List[ClassMetadata]
    functions: List[FunctionMetadata]

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.classes = []
        self.module_docstring = None
        logger.debug("Initialized CodeVisitor with empty function and class lists.")

    def visit_Module(self, node):
        self.module_docstring = ast.get_docstring(node)
        logger.debug(f"Extracted module docstring: {self.module_docstring}")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_name = node.name
        base_classes = [base.id for base in node.bases if isinstance(base, ast.Name)]
        methods = []
        docstring = ast.get_docstring(node)

        logger.info(f"Visiting class: {class_name}")
        logger.debug(f"Base classes: {base_classes}")
        logger.debug(f"Class docstring: {docstring}")

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self.extract_function_metadata(item))
                logger.debug(f"Extracted method: {item.name}")

        self.classes.append(ClassMetadata(
            name=class_name,
            base_classes=base_classes,
            methods=methods,
            docstring=docstring
        ))
        logger.info(f"Added class metadata for: {class_name}")

    def visit_FunctionDef(self, node):
        function_metadata = self.extract_function_metadata(node)
        self.functions.append(function_metadata)
        logger.info(f"Visiting function: {function_metadata.name}")
        logger.debug(f"Function metadata: {function_metadata}")

    def extract_function_metadata(self, node: ast.FunctionDef) -> FunctionMetadata:
        name = node.name
        args = [
            {'name': arg.arg, 'type': ast.unparse(arg.annotation) if arg.annotation else None}
            for arg in node.args.args
        ]
        return_type = ast.unparse(node.returns) if node.returns else None
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        docstring = ast.get_docstring(node)

        logger.debug(f"Extracting function metadata for: {name}")
        logger.debug(f"Arguments: {args}")
        logger.debug(f"Return type: {return_type}")
        logger.debug(f"Decorators: {decorators}")
        logger.debug(f"Docstring: {docstring}")

        return FunctionMetadata(
            name=name,
            args=args,
            return_type=return_type,
            decorators=decorators,
            docstring=docstring
        )

def extract_metadata_from_file(file_path: str) -> ExtractionResult:
    logger.info(f"Extracting metadata from file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        logger.debug(f"Read source code from {file_path}")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise

    try:
        tree = ast.parse(source_code)
        logger.debug("Parsed AST from source code.")
    except SyntaxError as e:
        logger.error(f"Syntax error while parsing {file_path}: {e}")
        raise

    visitor = CodeVisitor()
    visitor.visit(tree)

    module_name = file_path.split('/')[-1].replace('.py', '')
    module_metadata = ModuleMetadata(
        name=module_name,
        docstring=visitor.module_docstring
    )

    logger.info(f"Extracted metadata for module: {module_name}")
    return ExtractionResult(
        module_metadata=module_metadata,
        classes=visitor.classes,
        functions=visitor.functions
    )