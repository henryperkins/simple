"""Custom exceptions for the documentation generator."""


class ProcessingError(Exception):
    """Raised when processing of code or documentation fails."""


class ResponseParsingError(Exception):
    """Exception raised when there is an error parsing the AI response."""
    pass

class ValidationError(Exception):
    """Raised when validation of input or output fails."""


class ConnectionError(Exception):
    """Raised when connection to external services fails."""


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""


class ExtractionError(Exception):
    """Raised when code extraction fails."""


class IntegrationError(Exception):
    """Raised when integration of documentation fails."""


class LiveError(Exception):
    """Live error for real-time processing issues."""


class DocumentationError(Exception):
    """Raised when documentation generation fails."""


class WorkflowError(Exception):
    """Raised when a workflow operation fails."""


class TokenLimitError(Exception):
    """Raised when token limits are exceeded."""


class DocumentationGenerationError(Exception):
    """Raised when there is an error in generating documentation."""


class APICallError(Exception):
    """Raised when an API call fails."""
