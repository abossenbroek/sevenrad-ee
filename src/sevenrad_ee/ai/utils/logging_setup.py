"""
Centralized logging setup for GEPA optimization debugging.

This module provides comprehensive logging configuration with:
- Dual handlers (console INFO + file DEBUG)
- Context variable injection (request_id, phase)
- Structured logging for request tracing

Documentation Type: Technical Reference
Part of: Phase 3 - GEPA Optimization Debugging Infrastructure
"""

import contextvars
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)
phase_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "phase", default="setup"
)


class ContextualFilter(logging.Filter):
    """Inject request_id and phase from contextvars into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context variables to log record.

        Args:
            record: Log record to enhance

        Returns:
            Always True to pass the record through

        """
        setattr(record, "request_id", request_id_var.get("N/A"))
        setattr(record, "phase", phase_var.get("setup"))
        return True


def setup_logging(
    log_file: str | Path = "gepa_optimization_debug.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Configure logging for both console and file output.

    Creates two handlers:
    - Console: INFO level for user-friendly real-time feedback
    - File: DEBUG level for comprehensive debugging details

    Args:
        log_file: Path to log file (default: gepa_optimization_debug.log)
        console_level: Logging level for console (default: INFO)
        file_level: Logging level for file (default: DEBUG)

    Example:
        >>> from sevenrad_ee.ai.utils.logging_setup import setup_logging
        >>> setup_logging(log_file="my_debug.log")
        >>> # All loggers will now write to both console and file

    """
    # Remove existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger to DEBUG (handlers will filter based on their levels)
    root_logger.setLevel(logging.DEBUG)

    # Create formatter with custom fields for request_id and phase
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(phase)s:%(request_id)s] - %(message)s"
    )
    formatter = logging.Formatter(log_format)

    # Console handler (INFO level for user-friendly output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG level for complete details)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add contextual filter to all handlers
    contextual_filter = ContextualFilter()
    for handler in root_logger.handlers:
        handler.addFilter(contextual_filter)

    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized: console={logging.getLevelName(console_level)}, "
        f"file={logging.getLevelName(file_level)}"
    )
    logger.info(f"Log file: {Path(log_file).absolute()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        >>> from sevenrad_ee.ai.utils.logging_setup import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This message includes request_id and phase automatically")

    """
    return logging.getLogger(name)
