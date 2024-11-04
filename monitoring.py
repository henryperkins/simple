# monitoring.py
import sentry_sdk
import logging
from sentry_sdk.integrations.logging import LoggingIntegration

def initialize_sentry(environment="development", traces_sample_rate=1.0):
    """Initialize Sentry SDK with appropriate configuration.
    
    Args:
        environment (str): The environment name (development, production, etc.)
        traces_sample_rate (float): Sample rate for performance monitoring
    """
    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR
    )
    
    sentry_sdk.init(
        dsn="https://04fe86fc2c7757166d62bf0f2e1745c7@o4508070823395328.ingest.us.sentry.io/4508236588122112",
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[sentry_logging],
        attach_stacktrace=True,
        send_default_pii=False
    )

def capture_openai_error(error, context=None):
    """Capture OpenAI-related errors with context.
    
    Args:
        error (Exception): The error to capture
        context (dict, optional): Additional context information
    """
    with sentry_sdk.push_scope() as scope:
        if context:
            scope.set_context("openai_context", context)
        sentry_sdk.capture_exception(error)