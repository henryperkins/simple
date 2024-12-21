"""
Code extraction package for analyzing Python source code.
"""

from typing import Any, Optional
from core.logger import LoggerSetup
from core.metrics import Metrics
from core.docstring_processor import DocstringProcessor
from core.dependency_injection import Injector

# Import extractors
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.code_extractor import CodeExtractor
from core.extraction.extraction_utils import (
    extract_decorators,
    extract_attributes,
    extract_instance_attributes,
    extract_bases,
    get_node_name,
)

logger = LoggerSetup.get_logger(__name__)


async def initialize_extractors(
    config: Any = None, correlation_id: Optional[str] = None
) -> None:
    """Initialize extraction system with dependencies."""
    try:
        metrics = Metrics(correlation_id=correlation_id)
        docstring_processor = DocstringProcessor()

        # Register core dependencies if not already registered
        if not Injector.is_registered("metrics_calculator"):
            Injector.register("metrics_calculator", metrics)

        if not Injector.is_registered("docstring_processor"):
            Injector.register("docstring_processor", docstring_processor)

        logger.info("Extraction dependencies initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize extractors: {e}", exc_info=True)
        raise


__all__ = [
    "CodeExtractor",
    "ClassExtractor",
    "FunctionExtractor",
    "DependencyAnalyzer",
    "initialize_extractors",
    "extract_decorators",
    "extract_attributes",
    "extract_instance_attributes",
    "extract_bases",
    "get_node_name",
]
