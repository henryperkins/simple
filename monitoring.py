import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from config import Config
import logging

def initialize_sentry(environment="production", traces_sample_rate=1.0):
    """Initialize Sentry SDK with appropriate configuration."""
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
    )
    
    sentry_sdk.init(
        dsn=Config.SENTRY_DSN,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[sentry_logging],
        attach_stacktrace=True,
        send_default_pii=False
    )
    logging.info("Sentry initialized.")

def capture_openai_error(error, context=None):
    """Capture OpenAI-related errors with context."""
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_extra(key, value)
        sentry_sdk.capture_exception(error)