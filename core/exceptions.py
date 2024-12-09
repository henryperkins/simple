"""Custom exceptions for the documentation generator."""

class ProcessingError(Exception):
    """Raised when processing of code or documentation fails."""
    pass

class ValidationError(Exception):
    """Raised when validation of input or output fails."""
    pass

class ConnectionError(Exception):
    """Raised when connection to external services fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass

class ExtractionError(Exception):
    """Raised when code extraction fails."""
    pass

class IntegrationError(Exception):
    """Raised when integration of documentation fails."""
    pass

class DocumentationError(Exception):
    """Raised when documentation generation fails."""
    pass

class WorkflowError(Exception):
    """Raised when a workflow operation fails."""
    pass

class TokenLimitError(Exception):
    """Raised when token limits are exceeded."""
    pass
