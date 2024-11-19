"""
Custom Exceptions Module

This module defines custom exceptions used in the project.

Version: 1.0.0
Author: Development Team
"""

class TooManyRetriesError(Exception):
    """Exception raised when the maximum number of retries is exceeded."""
    def __init__(self, message: str = "Maximum retry attempts exceeded"):
        self.message = message
        super().__init__(self.message)
        
class DocumentationError(Exception):
    """Base class for documentation-related errors."""
    pass

class ValidationError(DocumentationError):
    """Raised when documentation validation fails."""
    pass

class ExtractionError(DocumentationError):
    """Raised when code extraction fails."""
    pass

class AIGenerationError(DocumentationError):
    """Raised when AI generation fails."""
    pass
    
class CacheError(Exception):
    """Exception raised for cache-related errors."""
    pass
    
class AIServiceError(Exception):
    """Base exception for AI service errors."""
    pass

class ValidationError(AIServiceError):
    """Validation related errors."""
    pass

class ProcessingError(AIServiceError):
    """Processing related errors."""
    pass