import sentry_sdk
import os
from core.logger import LoggerSetup
from dotenv import load_dotenv

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("monitoring")

# Load environment variables from .env file
load_dotenv()

# Load environment variables and validate
sentry_dsn = os.getenv("SENTRY_DSN")

if not sentry_dsn:
    logger.error("SENTRY_DSN is not set.")
    raise ValueError("SENTRY_DSN is not set.")

def initialize_sentry():
    try:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
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
