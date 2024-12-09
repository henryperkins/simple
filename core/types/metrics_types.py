"""Types for metrics calculations."""
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class MetricData:
    """Container for code metrics."""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_metrics: Dict[str, float] = field(default_factory=dict)
    lines_of_code: int = 0
    complexity_graph: Optional[str] = None
    total_functions: int = 0
    scanned_functions: int = 0
    total_classes: int = 0
    scanned_classes: int = 0
    
    @property
    def function_scan_ratio(self) -> float:
        """Calculate the ratio of successfully scanned functions."""
        return self.scanned_functions / self.total_functions if self.total_functions > 0 else 0.0
    
    @property
    def class_scan_ratio(self) -> float:
        """Calculate the ratio of successfully scanned classes."""
        return self.scanned_classes / self.total_classes if self.total_classes > 0 else 0.0
