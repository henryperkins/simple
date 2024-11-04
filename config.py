# config.py

import os
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

# Environment variables for API keys and endpoints
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # Add this line
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
            "api-key": AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json"
        }
    elif service == "openai":
        return {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    else:
        raise ValueError("Invalid service specified")

def get_azure_endpoint():
    """Construct the Azure OpenAI endpoint URL.

    Returns:
        str: The constructed endpoint URL.
    """
    return f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/models"