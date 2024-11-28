class WorkflowError(Exception):
    """Base exception class for workflow-related errors."""
    
    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)

# exceptions.py  
  
class ConfigurationError(Exception):  
    """Exception raised for errors in the configuration."""  
    pass  
  
class ProcessingError(Exception):  
    """Exception raised for errors during processing."""  
    pass  
  
class ValidationError(Exception):  
    """Exception raised for validation errors."""  
    pass  
  
class APIError(Exception):  
    """Exception raised for API-related errors."""  
    pass  
  
class CacheError(Exception):  
    """Exception raised for cache-related errors."""  
    pass  
  
class TokenLimitError(Exception):  
    """Exception raised when token limits are exceeded."""  
    pass  

class ConfigurationError(WorkflowError):
    """Raised when there are configuration-related issues."""
    pass


class AIInteractionError(WorkflowError):
    """Raised when there are issues with AI service interactions."""
    pass


class CacheError(WorkflowError):
    """Raised when there are caching-related issues."""
    pass


class DocumentationError(WorkflowError):
    """Raised when there are documentation generation issues."""
    pass


class AIServiceError(WorkflowError):
    """Raised when there are issues with the AI service."""
    pass


class TokenLimitError(WorkflowError):
    """Raised when token limits are exceeded."""
    pass


class ValidationError(WorkflowError):
    """Raised when validation fails."""
    pass


class ProcessingError(WorkflowError):
    """Raised when processing fails."""
    pass


class TooManyRetriesError(WorkflowError):
    """Raised when too many retries have been attempted."""
    pass
