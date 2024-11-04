# monitoring.py
import sentry_sdk
import logging
from sentry_sdk.integrations.logging import LoggingIntegration

def initialize_sentry(environment="development", traces_sample_rate=1.0):
    """Initialize Sentry SDK with appropriate configuration."""
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
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
    """Capture OpenAI-related errors with context."""
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_extra(key, value)
        sentry_sdk.capture_exception(error)