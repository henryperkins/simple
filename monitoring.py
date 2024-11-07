import sentry_sdk
from core.logger import LoggerSetup
from core.config.settings import Settings

def initialize_sentry():
    logger = LoggerSetup.get_logger("monitoring")
    try:
        settings = Settings()
        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        raise

def capture_exception(exception: Exception):
    """
    Capture and report an exception to Sentry.

    Args:
        exception (Exception): The exception to capture.
    """
    try:
        sentry_sdk.capture_exception(exception)
    except Exception as e:
        logger.error(f"Failed to capture exception: {e}")