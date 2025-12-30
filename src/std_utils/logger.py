import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "comini",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "comini") -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name (use dot notation for hierarchy, e.g., "comini.rag")
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, set up with defaults
    if not logger.handlers and not logger.parent.handlers:
        setup_logger(name)
    
    return logger

