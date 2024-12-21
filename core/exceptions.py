"""
Custom exception classes for the AI documentation service.

This module defines a hierarchy of custom exceptions used throughout the application.
These exceptions provide a structured way to handle errors and improve the clarity
of error messages and logging.
"""


class WorkflowError(Exception):
    """
    Base exception class for workflow-related errors.

    This is a generic exception for any error that occurs during the overall
    workflow of the application.
    """

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class ConfigurationError(WorkflowError):
    """
    Exception raised for errors in the application's configuration.

    This exception is used when there are issues with loading, parsing, or
    validating the application's configuration settings.
    """

    pass


class ProcessingError(WorkflowError):
    """
    Exception raised for errors during general processing.

    This exception is used for errors that occur during the processing of data,
    such as parsing, formatting, or other data manipulation tasks.
    """

    pass


class DocumentationError(WorkflowError):
    """
    Exception raised for errors during documentation generation.

    This exception is used when there are issues with generating documentation,
    including errors in code extraction, AI interaction, or markdown generation.
    """

    pass


class ValidationError(WorkflowError):
    """
    Exception raised for validation errors.

    This exception is used when data fails validation against a defined schema
    or other validation rules.
    """

    pass


class APIError(WorkflowError):
    """
    Exception raised for API-related errors.

    This exception is used when there are issues with API calls, such as
    authentication failures, network errors, or invalid responses.
    """

    pass


class CacheError(WorkflowError):
    """
    Exception raised for cache-related errors.

    This exception is used when there are issues with the application's cache,
    such as errors loading, saving, or accessing cached data.
    """

    pass


class ExtractionError(WorkflowError):
    """
    Exception raised when code extraction fails.

    This exception is used when there are issues with extracting code elements
    from source code using the AST or other methods.
    """

    pass


class MetricsError(WorkflowError):
    """
    Exception raised for errors in metrics tracking.

    This exception is used when there are issues with tracking or reporting
    metrics data within the application.
    """

    pass


class InvalidRequestError(WorkflowError):
    """
    Exception raised for invalid requests.

    This exception is used when a request is invalid or malformed.
    """

    pass


class CircularDependencyError(WorkflowError):
    """
    Exception raised for circular dependencies.

    This exception is used when circular dependencies are detected in the code.
    """

    pass


class TokenLimitError(WorkflowError):
    """
    Exception raised when token limits are exceeded.

    This exception is used when the number of tokens in a prompt or completion
    exceeds the maximum allowed limit.
    """

    pass


class AIInteractionError(WorkflowError):
    """
    Raised when there are issues with AI service interactions.

    This exception is used when there are problems with the communication
    or data exchange with the AI service.
    """

    pass


class AIServiceError(WorkflowError):
    """
    Raised when there are issues with the AI service itself.

    This exception is used when there are internal errors within the AI service.
    """

    pass


class TooManyRetriesError(WorkflowError):
    """
    Raised when too many retries have been attempted.

    This exception is used when an operation has been retried too many times
    and has still failed.
    """

    pass


class APICallError(APIError):
    """
    Exception raised for errors during API calls.

    This exception is used when there are issues during the API call process,
    such as network errors, timeouts, or invalid responses.
    """

    pass


class DataValidationError(ValidationError):
    """
    Raised when data validation fails.

    This exception is used when data fails validation against a defined schema
    or other validation rules.
    """

    def __init__(self, message: str):
        super().__init__(message)


class ResponseParsingError(ProcessingError):
    """
    Exception raised when there is an error parsing the AI response.

    This exception is used when the response from the AI model cannot be
    parsed correctly.
    """

    pass


class IntegrationError(WorkflowError):
    """
    Raised when integration of documentation fails.

    This exception is used when there are issues integrating the generated
    documentation with other systems or processes.
    """

    pass


class LiveError(WorkflowError):
    """
    Live error for real-time processing issues.

    This exception is used for errors that occur during real-time processing,
    such as issues with the user interface or live updates.
    """

    pass


class DocumentationGenerationError(DocumentationError):
    """
    Raised when there is an error in generating documentation.

    This exception is used when there are specific errors during the
    documentation generation process.
    """

    pass


class ConnectionError(WorkflowError):
    """
    Raised when connection to external services fails.

    This exception is used when there are issues connecting to external
    services, such as databases or APIs.
    """

    pass


class PromptGenerationError(ProcessingError):
    """
    Exception raised for errors during prompt generation.

    This exception is used when there are issues with creating prompts for the AI,
    such as template rendering errors or invalid input data.
    """

    pass


class TemplateLoadingError(ProcessingError):
    """
    Exception raised for errors during template loading.

    This exception is used when there are issues with loading or rendering Jinja templates.
    """

    pass


class DependencyAnalysisError(ProcessingError):
    """
    Exception raised for errors during dependency analysis.

    This exception is used when there are issues with analyzing dependencies in the code.
    """

    pass


class MaintainabilityError(ProcessingError):
    """
    Exception raised for errors during maintainability calculation.

    This exception is used when there are issues with calculating the maintainability index.
    """

    pass
