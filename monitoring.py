# monitoring.py

import os
import logging  # Added import for logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from logging_utils import setup_logger
from config import Config

# Initialize a logger specifically for this module
logger = setup_logger("monitoring")


def initialize_sentry():
    """
    Initialize Sentry SDK for error tracking and monitoring.

    This function configures and initializes the Sentry SDK using the DSN provided
    in the environment variables. It also integrates Sentry with the Python logging
    module to capture logs at various levels.

    Raises:
        ValueError: If the SENTRY_DSN environment variable is not set.
        sentry_sdk.init: Propagates any exception raised during Sentry initialization.
    """
    try:
        sentry_dsn = Config.get_variable("SENTRY_DSN")
        if not sentry_dsn:
            logger.warning(
                "SENTRY_DSN is not set. Sentry will not be initialized. "
                "Consider setting the SENTRY_DSN environment variable for error tracking."
            )
            return

        # Configure Sentry logging integration
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send errors as events
        )

        # Initialize Sentry SDK with integrations
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[sentry_logging],
            traces_sample_rate=1.0,  # Adjust based on your performance monitoring needs
            environment=os.getenv("ENVIRONMENT", "production"),  # e.g., development, staging, production
            release=os.getenv("RELEASE_VERSION", "unknown"),     # Optional: Set release version
        )

        logger.info("Sentry has been initialized successfully.")

    except ValueError as ve:
        logger.error(f"Configuration error during Sentry initialization: {ve}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during Sentry initialization: {e}")
        sentry_sdk.capture_exception(e)
        raise


def capture_exception(exception: Exception):
    """
    Capture and report an exception to Sentry.

    Args:
        exception (Exception): The exception to capture.
    """
    try:
        sentry_sdk.capture_exception(exception)
        logger.debug("Captured exception and reported to Sentry.")
    except Exception as e:
        logger.error(f"Failed to capture exception with Sentry: {e}")


def capture_message(message: str, level: str = "error"):
    """
    Capture and report a custom message to Sentry.

    Args:
        message (str): The message to capture.
        level (str): The severity level of the message ('debug', 'info', 'warning', 'error', 'critical').

    Raises:
        ValueError: If an invalid logging level is provided.
    """
    valid_levels = {"debug", "info", "warning", "error", "critical"}
    if level not in valid_levels:
        logger.error(f"Invalid log level '{level}' provided to capture_message.")
        raise ValueError(f"Invalid log level '{level}'. Valid levels are: {valid_levels}")

    try:
        sentry_sdk.capture_message(message, level=level)
        logger.debug(f"Captured message to Sentry at level '{level}': {message}")
    except Exception as e:
        logger.error(f"Failed to capture message with Sentry: {e}")
