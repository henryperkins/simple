import os
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """Configuration settings for the application."""

    # OpenAI API configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Azure OpenAI configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    # Sentry configuration
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
    SENTRY_RELEASE = os.getenv("SENTRY_RELEASE", "1.0.0")

    # Cache configuration
    CACHE_MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", 500))

    @staticmethod
    def validate():
        """Validate the configuration settings."""
        required_vars = [
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "SENTRY_DSN"
        ]
        missing_vars = [var for var in required_vars if not getattr(Config, var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

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

def get_azure_endpoint():
    """Construct the Azure OpenAI endpoint URL.

    Returns:
        str: The constructed endpoint URL.
    """
    return f"{Config.AZURE_OPENAI_ENDPOINT}/openai/deployments/{Config.AZURE_OPENAI_DEPLOYMENT_NAME}/models"
