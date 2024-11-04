import os
<<<<<<< HEAD
from dotenv import load_dotenv

=======
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from dotenv import load_dotenv

# Load environment variables from a .env file
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
load_dotenv()

class Config:
    """Configuration settings for the application."""
<<<<<<< HEAD
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
=======

    # OpenAI API configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Azure OpenAI configuration
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
<<<<<<< HEAD
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
    SENTRY_RELEASE = os.getenv("SENTRY_RELEASE", "1.0.0")
=======

    # Sentry configuration
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
    SENTRY_RELEASE = os.getenv("SENTRY_RELEASE", "1.0.0")

    # Cache configuration
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
    CACHE_MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", 500))

    @staticmethod
    def validate():
<<<<<<< HEAD
        """Validate that all required environment variables are set."""
        required_vars = [
            "OPENAI_API_KEY",
            "OPENAI_MODEL_NAME",
=======
        """Validate the configuration settings."""
        required_vars = [
            "OPENAI_API_KEY",
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "SENTRY_DSN"
        ]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

<<<<<<< HEAD
def get_service_headers(service: str) -> dict:
    """
    Get the appropriate headers for the specified AI service.

    Args:
        service (str): The AI service to use ('azure' or 'openai').

    Returns:
        dict: A dictionary of headers.
    
    Raises:
        ValueError: If an invalid service is specified.
    """
=======
# Validate configuration on import
Config.validate()

def initialize_sentry():
    """Initialize Sentry with proper configuration."""
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),  # Use environment variable for DSN
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        release=os.getenv("SENTRY_RELEASE", "1.0.0"),
        traces_sample_rate=1.0,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            ),
            AsyncioIntegration(),
        ],
        before_send=before_send
    )

def before_send(event, hint):
    """Filter sensitive data before sending to Sentry."""
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        if isinstance(exc_value, (KeyError, ValueError)):
            event['fingerprint'] = ['input-validation-error']
    return event

def get_service_headers(service):
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
    if service == "azure":
        return {
            "api-key": Config.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json"
        }
    elif service == "openai":
        return {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    else:
        raise ValueError("Invalid service specified")

<<<<<<< HEAD
def get_azure_endpoint() -> str:
    """
    Construct the Azure OpenAI endpoint URL.
=======
def get_azure_endpoint():
    """Construct the Azure OpenAI endpoint URL.
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6

    Returns:
        str: The constructed endpoint URL.
    """
<<<<<<< HEAD
    return f"{Config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={Config.AZURE_OPENAI_API_VERSION}"
=======
    return f"{Config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT_NAME}/models"
>>>>>>> 2d94ecaaf1aebee3c0cef377a3f4a9dfad24c7e6
