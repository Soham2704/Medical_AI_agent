import logging
import sys

# Define the log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create a formatter
formatter = logging.Formatter(LOG_FORMAT)

# --- Setup File Handler (logs to a file) ---
# This will save all logs (DEBUG level and up) to 'chat_log.log'
file_handler = logging.FileHandler("chat_log.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# --- Setup Stream Handler (logs to the terminal) ---
# This will print INFO level logs and up to your terminal
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

def get_logger(name):
    """
    Sets up and returns a logger with both file and stream handlers.
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # Set the overall level to the lowest (DEBUG) to allow handlers to filter
    logger.setLevel(logging.DEBUG)
    
    # Add handlers only if they haven't been added already
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger

# Create a main logger for our app
app_logger = get_logger("MedicalAIAssistant")