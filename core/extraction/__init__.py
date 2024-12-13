"""
Code extraction package for analyzing Python source code.
"""

from typing import Dict, Any, Optional

from core.logger import LoggerSetup
from core.metrics import Metrics
from core.docstring_processor import DocstringProcessor
from core.extraction.code_extractor import CodeExtractor
from core.extraction.class_extractor import ClassExtractor
from core.extraction.function_extractor import FunctionExtractor
from core.extraction.dependency_analyzer import DependencyAnalyzer
from core.types.base import Injector

logger = LoggerSetup.get_logger(__name__)


def setup_extractors(
    metrics: Optional[Metrics] = None,
    docstring_processor: Optional[DocstringProcessor] = None,
) -> None:
    """Setup extraction dependencies."""
    try:
        if not Injector.is_registered("metrics_calculator"):
            Injector.register("metrics_calculator", metrics or Metrics())

        if not Injector.is_registered("docstring_parser"):
            Injector.register(
                "docstring_parser", docstring_processor or DocstringProcessor()
            )

        logger.info("Extraction dependencies initialized successfully")
    except Exception as e:
        logger.error(f"Failed to setup extractors: {e}", exc_info=True)
        raise


__all__ = [
    "CodeExtractor",
    "ClassExtractor",
    "FunctionExtractor",
    "DependencyAnalyzer",
    "setup_extractors",
]
