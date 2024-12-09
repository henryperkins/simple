class WorkflowError(Exception):
    """Base exception class for workflow-related errors."""

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


class ProcessingError(Exception):
    """Exception raised for errors during processing."""

    pass


class DocumentationError(Exception):
    """Exception raised for errors during documentation generation."""

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


class ExtractionError(Exception):
    """Exception raised when code extraction fails."""

    pass


class TokenLimitError(Exception):
    """Exception raised when token limits are exceeded."""

    pass


class AIInteractionError(WorkflowError):
    """Raised when there are issues with AI service interactions."""

    pass


class AIServiceError(WorkflowError):
    """Raised when there are issues with the AI service."""

    pass


class TooManyRetriesError(WorkflowError):
    """Raised when too many retries have been attempted."""

    pass
