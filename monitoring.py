import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from config import Config
from logging_utils import setup_logger

# Initialize a logger specifically for the monitoring module
logger = setup_logger("monitoring")

def initialize_sentry(environment="production", traces_sample_rate=1.0):
    """Initialize Sentry SDK with appropriate configuration."""
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
    )
    
    try:
        sentry_sdk.init(
            dsn=Config.SENTRY_DSN,
            environment=environment,
            traces_sample_rate=traces_sample_rate,
            integrations=[sentry_logging],
            attach_stacktrace=True,
            send_default_pii=False
        )
        logger.info("Sentry initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")

def capture_openai_error(error, context=None):
    """Capture OpenAI-related errors with context."""
    logger.error(f"Capturing OpenAI error: {error}")
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_extra(key, value)
                logger.debug(f"Set context for Sentry: {key} = {value}")
        sentry_sdk.capture_exception(error)
    logger.info("Error captured in Sentry.")
