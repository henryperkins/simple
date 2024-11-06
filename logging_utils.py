# logging_utils.py

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Configure basic logging first, before any module can use it
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define main log directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure logs directory exists
ROTATION_DIR = os.path.join(LOG_DIR, "archive", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Initialize root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

def setup_logger(module_name: str) -> logging.Logger:
    """Sets up a logger for a specific module with file rotation."""
    # Ensure module log directory exists
    module_log_dir = os.path.join(LOG_DIR, module_name)
    os.makedirs(module_log_dir, exist_ok=True)

    # Define log file path
    log_file_path = os.path.join(module_log_dir, f"{module_name}.log")

    # Rotate existing logs to archive if they exist
    if os.path.exists(log_file_path):
        os.makedirs(ROTATION_DIR, exist_ok=True)
        os.rename(
            log_file_path,
            os.path.join(
                ROTATION_DIR,
                f"{module_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            )
        )

    # Set up logger with rotation
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # Rotating file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10**6, backupCount=5)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    return logger


def generate_summary():
    """Generates a summary log file with critical information from all module logs."""
    logger = setup_logger("summary")
    summary_log_path = os.path.join(LOG_DIR, "summary.log")
    
    with open(summary_log_path, 'w') as summary_file:
        for module_name in os.listdir(LOG_DIR):
            module_log_path = os.path.join(LOG_DIR, module_name, f"{module_name}.log")
            if os.path.exists(module_log_path):
                with open(module_log_path, 'r') as log_file:
                    for line in log_file:
                        if "ERROR" in line or "WARNING" in line:
                            summary_file.write(line)
    
    logger.info(f"Summary log generated at: {summary_log_path}")


def test_logging():
    """Test function to verify logging is working."""
    logger = setup_logger("test")
    logger.debug("Debug test message")
    logger.info("Info test message")
    logger.warning("Warning test message")
    logger.error("Error test message")
    print("Logs written to:", os.path.join(LOG_DIR, "test", "test.log"))


if __name__ == "__main__":
    test_logging()
