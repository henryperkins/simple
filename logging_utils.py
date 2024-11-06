# logging_utils.py

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Define main log directory
LOG_DIR = "logs"
ROTATION_DIR = os.path.join(LOG_DIR, "archive", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def setup_logger(module_name: str, console_logging: bool = False) -> logging.Logger:
    """
    Sets up a logger for a specific module with file rotation and optional console output.

    Args:
        module_name (str): The name of the module for which to set up the logger.
        console_logging (bool): If True, also log to console.

    Returns:
        logging.Logger: Configured logger for the module.
    """
    # Ensure module log directory exists
    module_log_dir = os.path.join(LOG_DIR, module_name)
    os.makedirs(module_log_dir, exist_ok=True)

    # Define log file path
    log_file_path = os.path.join(module_log_dir, f"{module_name}.log")

    # Rotate existing logs to archive if they exist
    if os.path.exists(log_file_path):
        try:
            os.makedirs(ROTATION_DIR, exist_ok=True)
            rotated_log_name = f"{module_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            os.rename(log_file_path, os.path.join(ROTATION_DIR, rotated_log_name))
        except OSError as e:
            print(f"Error rotating log file {log_file_path}: {e}")

    # Set up logger with rotation
    logger = logging.getLogger(module_name)
    if not logger.handlers:  # Prevent adding duplicate handlers
        logger.setLevel(logging.DEBUG)  # Set the level to DEBUG or adjust as needed

        # Rotating file handler
        file_handler = RotatingFileHandler(log_file_path, maxBytes=10**6, backupCount=5)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Optional console handler
        if console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(console_handler)

    return logger

def generate_summary():
    """
    Generates a summary log file with critical information from all module logs.

    The summary includes ERROR and WARNING level logs from all module logs.
    """
    summary_log_path = os.path.join(LOG_DIR, "summary.log")
    try:
        with open(summary_log_path, 'w', encoding='utf-8') as summary_file:
            for module_name in os.listdir(LOG_DIR):
                module_log_dir = os.path.join(LOG_DIR, module_name)
                module_log_path = os.path.join(module_log_dir, f"{module_name}.log")
                if os.path.exists(module_log_path):
                    try:
                        with open(module_log_path, 'r', encoding='utf-8') as log_file:
                            for line in log_file:
                                if "ERROR" in line or "WARNING" in line:
                                    summary_file.write(line)
                    except OSError as e:
                        print(f"Error reading log file {module_log_path}: {e}")
        print(f"Summary log generated at: {summary_log_path}")
    except OSError as e:
        print(f"Error writing summary log file: {e}")