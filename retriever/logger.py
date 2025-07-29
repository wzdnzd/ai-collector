#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standardized logging system for the async pipeline project.
Based on gemini-balance logger with modular organization.

Features:
- Color-coded console output with ANSI support
- File logging with rotation (main log + error-only log)
- API key redaction for security
- Module-specific loggers
- Automatic log cleanup
- Fixed-width formatting for readability

Log files are saved to the 'logs' directory:
- pipeline_YYYYMMDD_HHMMSS.log: All log levels
- pipeline_errors_YYYYMMDD_HHMMSS.log: ERROR and CRITICAL only
"""

import atexit
import ctypes
import logging
import logging.handlers
import platform
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from models import LogFileInfo, LoggingStats

# ANSI color codes for different log levels
COLORS = {
    "DEBUG": "\033[34m",  # Blue
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[1;31m",  # Bold Red
}

# Enable ANSI support on Windows
if platform.system() == "Windows":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


class ColoredFormatter(logging.Formatter):
    """Custom log formatter with color support and file location"""

    def format(self, record: logging.LogRecord) -> str:
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg=record.msg,
            args=record.args,
            exc_info=record.exc_info,
            func=record.funcName,
            stack_info=getattr(record, "stack_info", None),
        )

        # Copy additional attributes
        for key, value in record.__dict__.items():
            if not hasattr(record_copy, key):
                setattr(record_copy, key, value)

        # Get color code for log level
        color = COLORS.get(record_copy.levelname, "")
        # Add color code and reset code
        record_copy.levelname = f"{color}{record_copy.levelname}\033[0m"
        # Create fixed-width string with filename and line number
        record_copy.fileloc = f"[{record_copy.filename}:{record_copy.lineno}]"
        return super().format(record_copy)


class APIKeyRedactionFormatter(logging.Formatter):
    """Custom formatter that redacts API keys in log messages"""

    # API key patterns to match
    API_KEY_PATTERNS = [
        r"\bAIza[0-9A-Za-z_-]{35}",  # Google API keys (Gemini)
        r"\bsk-[0-9A-Za-z_-]{20,}",  # OpenAI and sk- prefixed keys
        r"\bsk-proj-[0-9A-Za-z_-]{20,}",  # OpenAI project keys
        r"\banthrop[0-9A-Za-z_-]{20,}",  # Anthropic keys
        r"\bgsk_[0-9A-Za-z_-]{20,}",  # GooeyAI keys
        r"\bstab_[0-9A-Za-z_-]{20,}",  # StabilityAI keys
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Compile regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.API_KEY_PATTERNS]

    def format(self, record: logging.LogRecord) -> str:
        # Format the record normally first
        formatted_msg = super().format(record)
        # Redact API keys in the formatted message
        return self._redact_api_keys_in_message(formatted_msg)

    def _redact_api_keys_in_message(self, message: str) -> str:
        """Replace API keys in log message with redacted versions"""
        try:
            for pattern in self.compiled_patterns:

                def replace_key(match):
                    key = match.group(0)
                    return self._redact_key_for_logging(key)

                message = pattern.sub(replace_key, message)
            return message
        except Exception as e:
            # Log the error but don't expose the original message
            logger = logging.getLogger(__name__)
            logger.error(f"Error redacting API keys in log: {e}")
            return "[LOG_REDACTION_ERROR]"

    def _redact_key_for_logging(self, key: str) -> str:
        """Redact API key for safe logging (show first 6 and last 6 characters)"""
        if len(key) <= 12:
            return "*" * len(key)
        return f"{key[:6]}...{key[-6:]}"


# Log format with file location and fixed width
FORMATTER = ColoredFormatter("%(asctime)s | %(levelname)-17s | %(fileloc)-30s | %(message)s")


# File formatter without colors for file output
class FileFormatter(logging.Formatter):
    """File formatter that removes ANSI color codes and uses clean formatting"""

    def format(self, record: logging.LogRecord) -> str:
        # Create file location string
        record.fileloc = f"[{record.filename}:{record.lineno}]"
        return super().format(record)


class FileFormatterWithRedaction(logging.Formatter):
    """File formatter that combines clean formatting with API key redaction"""

    # API key patterns to match
    API_KEY_PATTERNS = [
        r"\bAIza[0-9A-Za-z_-]{35}",  # Google API keys (Gemini)
        r"\bsk-[0-9A-Za-z_-]{20,}",  # OpenAI and sk- prefixed keys
        r"\bsk-proj-[0-9A-Za-z_-]{20,}",  # OpenAI project keys
        r"\banthrop[0-9A-Za-z_-]{20,}",  # Anthropic keys
        r"\bgsk_[0-9A-Za-z_-]{20,}",  # GooeyAI keys
        r"\bstab_[0-9A-Za-z_-]{20,}",  # StabilityAI keys
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Compile regex patterns for better performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.API_KEY_PATTERNS]

    def format(self, record: logging.LogRecord) -> str:
        # Create file location string
        record.fileloc = f"[{record.filename}:{record.lineno}]"
        # Format the record normally first
        formatted_msg = super().format(record)
        # Then apply API key redaction
        return self._redact_api_keys_in_message(formatted_msg)

    def _redact_api_keys_in_message(self, message: str) -> str:
        """Replace API keys in log message with redacted versions"""
        try:
            for pattern in self.compiled_patterns:

                def replace_key(match):
                    key = match.group(0)
                    return self._redact_key_for_logging(key)

                message = pattern.sub(replace_key, message)
            return message
        except Exception as e:
            # Return original message if redaction fails
            return message

    def _redact_key_for_logging(self, key: str) -> str:
        """Redact API key for safe logging (show first 6 and last 6 characters)"""
        if len(key) <= 12:
            return "*" * len(key)
        return f"{key[:6]}...{key[-6:]}"


FILE_FORMATTER = FileFormatterWithRedaction("%(asctime)s | %(levelname)-8s | %(fileloc)-30s | %(message)s")

# Log level mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class Logger:
    """Centralized logger management system"""

    _loggers: Dict[str, logging.Logger] = {}
    _default_level = logging.INFO
    _logs_dir = None
    _file_handler = None
    _error_file_handler = None

    @staticmethod
    def _setup_file_handlers():
        """Setup file handlers for logging to files"""
        if Logger._file_handler is not None:
            return  # Already setup

        # Create logs directory
        Logger._logs_dir = Path("logs")
        Logger._logs_dir.mkdir(exist_ok=True)

        # Generate log file names with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main log file (all levels)
        main_log_file = Logger._logs_dir / f"pipeline_{timestamp}.log"
        Logger._file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8"  # 50MB
        )
        Logger._file_handler.setFormatter(FILE_FORMATTER)
        Logger._file_handler.setLevel(logging.DEBUG)

        # Error log file (ERROR and CRITICAL only)
        error_log_file = Logger._logs_dir / f"pipeline_errors_{timestamp}.log"
        Logger._error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
        )
        Logger._error_file_handler.setFormatter(FILE_FORMATTER)
        Logger._error_file_handler.setLevel(logging.ERROR)

    @staticmethod
    def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Setup and get logger instance
        :param name: logger name
        :param level: log level (optional, uses default if not provided)
        :return: logger instance
        """
        # Use provided level or default
        if level:
            log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
        else:
            log_level = Logger._default_level

        if name in Logger._loggers:
            # If logger exists, update its level if needed
            existing_logger = Logger._loggers[name]
            if existing_logger.level != log_level:
                existing_logger.setLevel(log_level)
            return existing_logger

        logger_instance = logging.getLogger(name)
        logger_instance.setLevel(log_level)
        logger_instance.propagate = False

        # Setup file handlers if not already done
        Logger._setup_file_handlers()

        # Add console handler with colored formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        console_handler.setLevel(log_level)
        logger_instance.addHandler(console_handler)

        # Add file handlers (they already have their own formatters)
        if Logger._file_handler:
            logger_instance.addHandler(Logger._file_handler)
        if Logger._error_file_handler:
            logger_instance.addHandler(Logger._error_file_handler)

        Logger._loggers[name] = logger_instance
        return logger_instance

    @staticmethod
    def get_logger(name: str) -> Optional[logging.Logger]:
        """
        Get existing logger
        :param name: logger name
        :return: logger instance or None
        """
        return Logger._loggers.get(name)

    @staticmethod
    def update_log_levels(log_level: str) -> None:
        """Update all existing loggers to new log level"""
        log_level_str = log_level.lower()
        new_level = LOG_LEVELS.get(log_level_str, logging.INFO)
        Logger._default_level = new_level

        for logger_name, logger_instance in Logger._loggers.items():
            if logger_instance.level != new_level:
                logger_instance.setLevel(new_level)

    @staticmethod
    def set_default_level(level: str) -> None:
        """Set default log level for new loggers"""
        Logger._default_level = LOG_LEVELS.get(level.lower(), logging.INFO)

    @staticmethod
    def get_logs_directory() -> Optional[Path]:
        """Get the logs directory path"""
        return Logger._logs_dir

    @staticmethod
    def cleanup_old_logs(days: int = 7) -> None:
        """Clean up log files older than specified days"""
        if not Logger._logs_dir or not Logger._logs_dir.exists():
            return

        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for log_file in Logger._logs_dir.glob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    print(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                print(f"Failed to clean up log file {log_file}: {e}")

    @staticmethod
    def flush_all_handlers():
        """Flush all file handlers to ensure logs are written"""
        if Logger._file_handler:
            Logger._file_handler.flush()
        if Logger._error_file_handler:
            Logger._error_file_handler.flush()

    @staticmethod
    def get_log_files_info() -> Dict[str, Dict[str, str]]:
        """Get detailed information about current log files"""
        info = {}
        if Logger._logs_dir and Logger._logs_dir.exists():
            for log_file in Logger._logs_dir.glob("*.log"):
                try:
                    stat = log_file.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    modified_time = datetime.fromtimestamp(stat.st_mtime)

                    info[log_file.name] = {
                        "size": f"{size_mb:.2f} MB",
                        "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "path": str(log_file.absolute()),
                    }
                except Exception as e:
                    info[log_file.name] = {"size": "Unknown", "modified": "Unknown", "error": str(e)}
        return info

    @staticmethod
    def get_log_stats() -> LoggingStats:
        """Get logging system statistics"""
        log_files_info = Logger.get_log_files_info()
        log_files = {}

        # Convert dict to LogFileInfo objects
        for filename, info in log_files_info.items():
            if isinstance(info, dict):
                log_files[filename] = LogFileInfo(
                    filename=filename,
                    size=info.get("size", ""),
                    modified=info.get("modified", ""),
                    path=info.get("path", ""),
                    error=info.get("error"),
                )

        return LoggingStats(
            active_loggers=len(Logger._loggers),
            log_files=log_files,
            logs_directory=str(Logger._logs_dir) if Logger._logs_dir else None,
        )


# Initialize logging system
def init_logging(level: str = "INFO", cleanup_days: int = 7):
    """
    Initialize logging system with specified level
    :param level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param cleanup_days: Number of days to keep old log files (default: 7)
    """
    Logger.set_default_level(level)

    # Clean up old log files
    Logger.cleanup_old_logs(cleanup_days)

    # Setup file handlers
    Logger._setup_file_handlers()

    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers()

    # Log initialization message
    main_logger = get_main_logger()
    main_logger.info(f"Logging system initialized - Level: {level.upper()}")
    main_logger.info(f"Logs directory: {Logger.get_logs_directory()}")
    main_logger.info(f"Log cleanup: keeping files for {cleanup_days} days")


def _setup_signal_handlers():
    """Setup signal handlers for graceful logging shutdown"""

    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        shutdown_logging()

    # Register signal handlers (only if not already registered)
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (ValueError, OSError):
        # Signal handling might not be available in some environments
        pass

    # Register exit handler
    atexit.register(shutdown_logging)


# Module-specific logger functions
def get_main_logger() -> logging.Logger:
    return Logger.setup_logger("main")


def get_config_logger() -> logging.Logger:
    return Logger.setup_logger("config")


def get_pipeline_logger() -> logging.Logger:
    return Logger.setup_logger("pipeline")


def get_search_logger() -> logging.Logger:
    return Logger.setup_logger("search")


def get_collect_logger() -> logging.Logger:
    return Logger.setup_logger("collect")


def get_check_logger() -> logging.Logger:
    return Logger.setup_logger("check")


def get_models_logger() -> logging.Logger:
    return Logger.setup_logger("models")


def get_task_manager_logger() -> logging.Logger:
    return Logger.setup_logger("task_manager")


def get_queue_manager_logger() -> logging.Logger:
    return Logger.setup_logger("queue_manager")


def get_result_manager_logger() -> logging.Logger:
    return Logger.setup_logger("result_manager")


def get_rate_limiter_logger() -> logging.Logger:
    return Logger.setup_logger("rate_limiter")


def get_load_balancer_logger() -> logging.Logger:
    return Logger.setup_logger("load_balancer")


def get_monitoring_logger() -> logging.Logger:
    return Logger.setup_logger("monitoring")


def get_application_logger() -> logging.Logger:
    return Logger.setup_logger("application")


def get_engine_logger():
    return Logger.setup_logger("engine")


def get_client_logger():
    return Logger.setup_logger("client")


def get_utils_logger():
    return Logger.setup_logger("utils")


# Provider logger
def get_provider_logger():
    return Logger.setup_logger("provider")


# Utility functions for log management
def get_current_log_files():
    """Get detailed information about current log files"""
    return Logger.get_log_files_info()


def get_logging_stats() -> LoggingStats:
    """Get comprehensive logging system statistics"""
    return Logger.get_log_stats()


def cleanup_logs(days: int = 7):
    """Clean up log files older than specified days"""
    Logger.cleanup_old_logs(days)


def flush_logs():
    """Flush all log handlers to ensure data is written to files"""
    Logger.flush_all_handlers()


def shutdown_logging():
    """Gracefully shutdown logging system and close all handlers"""
    Logger.flush_all_handlers()

    # Close file handlers
    if Logger._file_handler:
        Logger._file_handler.close()
    if Logger._error_file_handler:
        Logger._error_file_handler.close()

    # Clear logger cache
    Logger._loggers.clear()

    # Reset handlers
    Logger._file_handler = None
    Logger._error_file_handler = None


def setup_access_logging():
    """
    Configure access logging with API key redaction

    This function sets up custom access log formatting that automatically
    redacts API keys in HTTP access logs. It works by:

    1. Intercepting access log messages
    2. Using regex patterns to find API keys in URLs
    3. Replacing them with redacted versions (first6...last6)

    Supported API key formats:
    - Google/Gemini API keys: AIza[35 chars]
    - OpenAI API keys: sk-[48 chars]
    - OpenAI project keys: sk-proj-[48 chars]
    - Anthropic keys: anthrop[20+ chars]
    - GooeyAI keys: gsk_[20+ chars]
    - StabilityAI keys: stab_[20+ chars]
    """
    # Get access logger (if using uvicorn or similar)
    access_logger = logging.getLogger("uvicorn.access")

    # Remove existing handlers to avoid duplicate logs
    for handler in access_logger.handlers[:]:
        access_logger.removeHandler(handler)

    # Create new handler with API key redaction formatter
    handler = logging.StreamHandler(sys.stdout)
    access_formatter = APIKeyRedactionFormatter("%(asctime)s | %(levelname)-8s | %(message)s")
    handler.setFormatter(access_formatter)

    # Add the handler to access logger
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False

    return access_logger


if __name__ == "__main__":
    # Test the logging system
    init_logging("DEBUG")

    # Test different loggers
    main_logger = get_main_logger()
    config_logger = get_config_logger()
    search_logger = get_search_logger()

    main_logger.info("Main logger test - this should appear in console and file")
    config_logger.debug("Config logger test - debug level")
    search_logger.warning("Search logger test - warning level")

    # Test error logging (should go to both main log and error log)
    main_logger.error("Test error message - should appear in error log too")

    # Test API key redaction
    test_logger = get_provider_logger()
    test_logger.info("Testing API key redaction: sk-1234567890abcdefghij1234567890abcdefghij")
    test_logger.info("Testing Gemini key: AIza1234567890abcdefghij1234567890abcde")

    # Flush all handlers to ensure logs are written
    Logger.flush_all_handlers()

    # Show logging statistics
    stats = get_logging_stats()
    print(f"\n=== Logging System Statistics ===")
    print(f"Active loggers: {stats['active_loggers']}")
    print(f"Logger names: {', '.join(stats['logger_names'])}")
    print(f"Logs directory: {stats['logs_directory']}")
    print(f"File handlers active: {stats['file_handlers_active']}")

    print(f"\n=== Log Files ===")
    for filename, info in stats["log_files"].items():
        if "error" not in info:
            print(f"  - {filename}")
            print(f"    Size: {info['size']}")
            print(f"    Modified: {info['modified']}")
        else:
            print(f"  - {filename} (Error: {info['error']})")
            print(f"  - {filename} (Error: {info['error']})")
